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
#include <boost/function.hpp>
#include <boost/thread/xtime.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/service.hpp"
#include "asio/service_type_id.hpp"

namespace asio { class completion_context; }

namespace asio {
namespace detail {

class timer_queue_service
  : public virtual service
{
public:
  // The service type id.
  static const service_type_id id;

  // The handler for when a timer expires.
  typedef boost::function0<void> timer_handler;

  // Schedule a timer to fire once at the given start_time. The id of the new
  // timer is returned so that it may be cancelled.
  int schedule_timer(void* owner, const boost::xtime& start_time,
      const timer_handler& handler, completion_context& context);

  // Schedule a timer to fire first after at the start time, and then every
  // interval until the timer is cancelled. The id of the new timer is
  // returned so that it may be cancelled.
  int schedule_timer(void* owner, const boost::xtime& start_time,
      const boost::xtime& interval, const timer_handler& handler,
      completion_context& context);

  // Cancel the timer with the given id.
  void cancel_timer(void* owner, int timer_id);

private:
  // Schedule a timer to fire first after at the start time, and then every
  // interval until the timer is cancelled. A zero interval means that the
  // timer will fire once only. The id of the new timer is returned so that it
  // may be cancelled.
  virtual int do_schedule_timer(void* owner, const boost::xtime& start_time,
      const boost::xtime& interval, const timer_handler& handler,
      completion_context& context) = 0;

  // Cancel the timer with the given id.
  virtual void do_cancel_timer(void* owner, int timer_id) = 0;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TIMER_QUEUE_SERVICE_HPP
