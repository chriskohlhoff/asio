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

#ifndef ASIO_TIMER_QUEUE_SERVICE_HPP
#define ASIO_TIMER_QUEUE_SERVICE_HPP

#include "asio/service.hpp"
#include "asio/service_type_id.hpp"
#include "asio/timer_queue.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// The timer_queue_service class is a base class for service implementations
/// that provide the functionality required by the timer_queue class.
class timer_queue_service
  : public virtual service
{
public:
  typedef timer_queue::timer_handler timer_handler;

  /// The service type id.
  static const service_type_id id;

  /// Schedule a timer to fire first after at the start time, and then every
  /// interval until the timer is cancelled. A zero interval means that the
  /// timer will fire once only. The id of the new timer is returned so that it
  /// may be cancelled.
  virtual int schedule_timer(timer_queue& queue,
      const boost::xtime& start_time, const boost::xtime& interval,
      const timer_handler& handler, completion_context& context) = 0;

  /// Cancel the timer with the given id.
  virtual void cancel_timer(timer_queue& queue, int timer_id) = 0;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TIMER_QUEUE_SERVICE_HPP
