//
// timer_queue_provider.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/timer_queue_provider.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/bind.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

timer_queue_provider::
timer_queue_provider(
    demuxer& d)
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
          &timer_queue_provider::expire_timers, this)));
}

timer_queue_provider::
~timer_queue_provider()
{
  boost::mutex::scoped_lock lock(mutex_);
  is_stopping_ = true;
  stop_condition_.notify_all();
  lock.unlock();

  thread_->join();
}

service*
timer_queue_provider::
do_get_service(
    const service_type_id& service_type)
{
  if (service_type == timer_queue_service::id)
    return this;
  return 0;
}

int
timer_queue_provider::
do_schedule_timer(
    void* owner,
    const boost::xtime& start_time,
    const boost::xtime& interval,
    const timer_handler& handler,
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

namespace
{
  struct dummy_completion_handler
  {
    void operator()() {}
  };
}

void
timer_queue_provider::
do_cancel_timer(
    void* owner,
    int timer_id)
{
  boost::mutex::scoped_lock lock(mutex_);

  id_to_timer_map::iterator iter = id_to_timer_.find(timer_id);
  if (iter != id_to_timer_.end() && iter->second->second.owner == owner)
  {
    timer_queue_.erase(iter->second);
    id_to_timer_.erase(iter);
    lock.unlock();
    demuxer_.operation_completed(dummy_completion_handler());
  }
}

void
timer_queue_provider::
expire_timers()
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

} // namespace detail
} // namespace asio
