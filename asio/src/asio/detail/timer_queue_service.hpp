//
// timer_queue_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_TIMER_QUEUE_SERVICE_HPP
#define ASIO_DETAIL_TIMER_QUEUE_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <map>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/thread/xtime.hpp>
#include <boost/thread.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/completion_context.hpp"

namespace asio {
namespace detail {

template <typename Demuxer>
class timer_queue_service
{
public:
  // The handler for when a timer expires.
  typedef boost::function0<void> timer_handler;

  // Constructor.
  timer_queue_service(Demuxer& d)
    : demuxer_(d),
      mutex_(),
      thread_(),
      is_stopping_(false),
      stop_condition_(),
      timer_queue_(),
      id_to_timer_(),
      next_timer_id_(1)
  {
    thread_.reset(new boost::thread(boost::bind(
            &timer_queue_service<Demuxer>::expire_timers, this)));
  }

  // Destructor.
  ~timer_queue_service()
  {
    boost::mutex::scoped_lock lock(mutex_);
    is_stopping_ = true;
    stop_condition_.notify_all();
    lock.unlock();

    thread_->join();
  }

  // Schedule a timer to fire once at the given start_time. The id of the new
  // timer is returned so that it may be cancelled.
  int schedule_timer(void* owner, const boost::xtime& start_time,
      const timer_handler& handler, completion_context& context)
  {
    boost::mutex::scoped_lock lock(mutex_);

    timer_event new_event;
    new_event.handler = handler;
    new_event.interval.sec = 0;
    new_event.interval.nsec = 0;
    new_event.context = &context;
    new_event.owner = owner;
    new_event.id = next_timer_id_++;
    id_to_timer_.insert(std::make_pair(new_event.id,
          timer_queue_.insert(std::make_pair(start_time, new_event))));

    demuxer_.operation_started();
    stop_condition_.notify_one();

    return new_event.id;
  }

  // Schedule a timer to fire first after at the start time, and then every
  // interval until the timer is cancelled. The id of the new timer is
  // returned so that it may be cancelled.
  int schedule_timer(void* owner, const boost::xtime& start_time,
      const boost::xtime& interval, const timer_handler& handler,
      completion_context& context)
  {
    boost::mutex::scoped_lock lock(mutex_);

    timer_event new_event;
    new_event.handler = handler;
    new_event.interval = interval;
    new_event.context = &context;
    new_event.owner = owner;
    new_event.id = next_timer_id_++;
    id_to_timer_.insert(std::make_pair(new_event.id,
          timer_queue_.insert(std::make_pair(start_time, new_event))));

    demuxer_.operation_started();
    stop_condition_.notify_one();

    return new_event.id;
  }

  struct dummy_completion_handler
  {
    void operator()() {}
  };

  // Cancel the timer with the given id.
  void cancel_timer(void* owner, int timer_id)
  {
    boost::mutex::scoped_lock lock(mutex_);

    typename id_to_timer_map::iterator iter = id_to_timer_.find(timer_id);
    if (iter != id_to_timer_.end() && iter->second->second.owner == owner)
    {
      timer_queue_.erase(iter->second);
      id_to_timer_.erase(iter);
      lock.unlock();
      demuxer_.operation_completed(dummy_completion_handler());
    }
  }

private:
  // Loop for expiring timers until it is time to shut down.
  void expire_timers()
  {
    boost::mutex::scoped_lock lock(mutex_);

    while (!is_stopping_)
    {
      if (timer_queue_.size())
      {
        stop_condition_.timed_wait(lock, timer_queue_.begin()->first);

        boost::xtime now;
        boost::xtime_get(&now, boost::TIME_UTC);
        if (timer_queue_.size()
            && boost::xtime_cmp(now, timer_queue_.begin()->first) >= 0)
        {
          boost::xtime old_start_time = timer_queue_.begin()->first;
          timer_event event = timer_queue_.begin()->second;
          timer_queue_.erase(timer_queue_.begin());
          id_to_timer_.erase(event.id);
          if (event.interval.sec || event.interval.nsec)
          {
            boost::xtime new_start_time;
            new_start_time.sec = old_start_time.sec + event.interval.sec;
            new_start_time.nsec = old_start_time.nsec + event.interval.nsec;
            new_start_time.sec += new_start_time.nsec / 1000000000;
            new_start_time.nsec = new_start_time.nsec % 1000000000;
            id_to_timer_.insert(std::make_pair(event.id,
                  timer_queue_.insert(std::make_pair(new_start_time, event))));
            demuxer_.operation_started();
          }

          demuxer_.operation_completed(event.handler, *event.context);
        }
      }
      else
      {
        stop_condition_.wait(lock);
      }
    }
  }

  // The demuxer that owns this provider.
  Demuxer& demuxer_;

  // Mutex to protect access to internal data.
  boost::mutex mutex_;

  // Worker thread for waiting for timers to expire.
  boost::scoped_ptr<boost::thread> thread_;

  // Flag to indicate that the worker thread should stop.
  bool is_stopping_;

  // Condition variable to indicate that the worker thread should stop.
  boost::condition stop_condition_;

  // Function object for comparing xtimes.
  struct xtime_less
  {
    bool operator()(const boost::xtime& xt1, const boost::xtime& xt2)
    {
      return boost::xtime_cmp(xt1, xt2) < 0;
    }
  };

  // Information about each timer event.
  struct timer_event
  {
    boost::function0<void> handler;
    boost::xtime interval;
    completion_context* context;
    void* owner;
    int id;
  };

  // Ordered collection of events.
  typedef std::multimap<boost::xtime, timer_event, xtime_less> timer_queue_map;
  timer_queue_map timer_queue_;

  // Mapping from timer id to timer event.
  typedef std::map<int, typename timer_queue_map::iterator> id_to_timer_map;
  id_to_timer_map id_to_timer_;

  // The next available timer id.
  int next_timer_id_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TIMER_QUEUE_SERVICE_HPP
