//
// timer_queue.hpp
// ~~~~~~~~~~~~~~~
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

#ifndef ASIO_TIMER_QUEUE_HPP
#define ASIO_TIMER_QUEUE_HPP

#include <boost/function.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/xtime.hpp>
#include "asio/completion_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

class demuxer;
class timer_queue_service;

/// The timer_queue class provides asynchronous timer functionality.
class timer_queue
  : private boost::noncopyable
{
public:
  /// Constructor.
  explicit timer_queue(demuxer& d);

  /// Destructor.
  ~timer_queue();

  /// The handler for when a timer expires.
  typedef boost::function0<void> timer_handler;

  /// Schedule a timer to fire once at the given start_time. The id of the new
  /// timer is returned so that it may be cancelled.
  int schedule_timer(const boost::xtime& start_time,
      const timer_handler& handler,
      completion_context& context = completion_context::null());

  /// Schedule a timer to fire first after at the start time, and then every
  /// interval until the timer is cancelled. The id of the new timer is
  /// returned so that it may be cancelled.
  int schedule_timer(const boost::xtime& start_time,
      const boost::xtime& interval, const timer_handler& handler,
      completion_context& context = completion_context::null());

  /// Cancel the timer with the given id.
  void cancel_timer(int timer_id);

private:
  /// The backend service implementation.
  timer_queue_service& service_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TIMER_QUEUE_HPP
