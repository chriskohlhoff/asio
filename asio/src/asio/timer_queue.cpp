//
// timer_queue.cpp
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

#include "asio/timer_queue.hpp"
#include "asio/timer_queue_service.hpp"
#include "asio/demuxer.hpp"

namespace asio {

timer_queue::
timer_queue(
    demuxer& d)
  : service_(dynamic_cast<timer_queue_service&>(
        d.get_service(timer_queue_service::id)))
{
}

timer_queue::
~timer_queue()
{
}

int
timer_queue::
schedule_timer(
    const boost::xtime& start_time,
    const timer_handler& handler,
    completion_context& context)
{
  boost::xtime interval;
  interval.sec = 0;
  interval.nsec = 0;
  return service_.schedule_timer(*this, start_time, interval, handler,
      context);
}

int
timer_queue::
schedule_timer(
    const boost::xtime& start_time,
    const boost::xtime& interval,
    const timer_handler& handler,
    completion_context& context)
{
  return service_.schedule_timer(*this, start_time, interval, handler,
      context);
}

void
timer_queue::
cancel_timer(
    int timer_id)
{
  service_.cancel_timer(*this, timer_id);
}

} // namespace asio
