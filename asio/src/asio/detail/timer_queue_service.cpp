//
// timer_queue_service.cpp
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

#include "asio/detail/timer_queue_service.hpp"

namespace asio {
namespace detail {

const service_type_id timer_queue_service::id;

int
timer_queue_service::
schedule_timer(
    void* owner,
    const boost::xtime& start_time,
    const timer_handler& handler,
    completion_context& context)
{
  boost::xtime interval;
  interval.sec = 0;
  interval.nsec = 0;
  do_schedule_timer(owner, start_time, interval, handler, context);
}

int
timer_queue_service::
schedule_timer(
    void* owner,
    const boost::xtime& start_time,
    const boost::xtime& interval,
    const timer_handler& handler,
    completion_context& context)
{
  do_schedule_timer(owner, start_time, interval, handler, context);
}

void
timer_queue_service::
cancel_timer(
    void* owner,
    int timer_id)
{
  do_cancel_timer(owner, timer_id);
}

} // namespace detail
} // namespace asio
