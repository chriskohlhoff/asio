//
// timer_queue_provider.hpp
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

#ifndef ASIO_DETAIL_TIMER_QUEUE_PROVIDER_HPP
#define ASIO_DETAIL_TIMER_QUEUE_PROVIDER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <map>
#include <boost/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/demuxer.hpp"
#include "asio/completion_context.hpp"
#include "asio/service_provider.hpp"
#include "asio/detail/timer_queue_service.hpp"

namespace asio {
namespace detail {

class timer_queue_provider
  : public service_provider,
    public timer_queue_service
{
public:
  // Constructor.
  timer_queue_provider(demuxer& d);

  // Destructor.
  virtual ~timer_queue_provider();

private:
  // Return the service interface corresponding to the given type.
  virtual service* do_get_service(const service_type_id& service_type);

  // Schedule a timer to fire first after at the start time, and then every
  // interval until the timer is cancelled. A zero interval means that the
  // timer will fire once only. The id of the new timer is returned so that it
  // may be cancelled.
  virtual int do_schedule_timer(void* owner,
      const boost::xtime& start_time, const boost::xtime& interval,
      const timer_handler& handler, completion_context& context);

  // Cancel the timer with the given id.
  virtual void do_cancel_timer(void* owner, int timer_id);

  // Loop for expiring timers until it is time to shut down.
  void expire_timers();

  // The demuxer that owns this provider.
  demuxer& demuxer_;

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
  typedef std::map<int, timer_queue_map::iterator> id_to_timer_map;
  id_to_timer_map id_to_timer_;

  // The next available timer id.
  int next_timer_id_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TIMER_QUEUE_PROVIDER_HPP
