//
// timer_queue_set.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TIMER_QUEUE_SET_HPP
#define ASIO_DETAIL_TIMER_QUEUE_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/timer_queue_base.hpp"

namespace asio {
namespace detail {

class timer_queue_set
{
public:
  // Constructor.
  timer_queue_set()
    : first_(0)
  {
  }

  // Add a timer queue to the set.
  void insert(timer_queue_base* q)
  {
    q->next_ = first_;
    first_ = q;
  }

  // Remove a timer queue from the set.
  void erase(timer_queue_base* q)
  {
    if (first_)
    {
      if (q == first_)
      {
        first_ = q->next_;
        q->next_ = 0;
        return;
      }

      for (timer_queue_base* p = first_; p->next_; p = p->next_)
      {
        if (p->next_ == q)
        {
          p->next_ = q->next_;
          q->next_ = 0;
          return;
        }
      }
    }
  }

  // Determine whether all queues are empty.
  bool all_empty() const
  {
    for (timer_queue_base* p = first_; p; p = p->next_)
      if (!p->empty())
        return false;
    return true;
  }

  // Get the wait duration in milliseconds.
  long wait_duration_msec(long max_duration) const
  {
    long min_duration = max_duration;
    for (timer_queue_base* p = first_; p; p = p->next_)
      min_duration = p->wait_duration_msec(min_duration);
    return min_duration;
  }

  // Get the wait duration in microseconds.
  long wait_duration_usec(long max_duration) const
  {
    long min_duration = max_duration;
    for (timer_queue_base* p = first_; p; p = p->next_)
      min_duration = p->wait_duration_usec(min_duration);
    return min_duration;
  }

  // Dequeue all ready timers.
  void get_ready_timers(op_queue<operation>& ops)
  {
    for (timer_queue_base* p = first_; p; p = p->next_)
      p->get_ready_timers(ops);
  }

  // Dequeue all timers.
  void get_all_timers(op_queue<operation>& ops)
  {
    for (timer_queue_base* p = first_; p; p = p->next_)
      p->get_all_timers(ops);
  }

private:
  timer_queue_base* first_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TIMER_QUEUE_SET_HPP
