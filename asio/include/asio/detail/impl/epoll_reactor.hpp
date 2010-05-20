//
// detail/impl/epoll_reactor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_EPOLL_REACTOR_HPP
#define ASIO_DETAIL_IMPL_EPOLL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#if defined(ASIO_HAS_EPOLL)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Add a new timer queue to the reactor.
template <typename Time_Traits>
void epoll_reactor::add_timer_queue(timer_queue<Time_Traits>& timer_queue)
{
  mutex::scoped_lock lock(mutex_);
  timer_queues_.insert(&timer_queue);
}

// Remove a timer queue from the reactor.
template <typename Time_Traits>
void epoll_reactor::remove_timer_queue(timer_queue<Time_Traits>& timer_queue)
{
  mutex::scoped_lock lock(mutex_);
  timer_queues_.erase(&timer_queue);
}

// Schedule a new operation in the given timer queue to expire at the
// specified absolute time.
template <typename Time_Traits>
void epoll_reactor::schedule_timer(timer_queue<Time_Traits>& timer_queue,
    const typename Time_Traits::time_type& time, timer_op* op, void* token)
{
  mutex::scoped_lock lock(mutex_);
  if (!shutdown_)
  {
    bool earliest = timer_queue.enqueue_timer(time, op, token);
    io_service_.work_started();
    if (earliest)
      update_timeout();
  }
}

// Cancel the timer operations associated with the given token. Returns the
// number of operations that have been posted or dispatched.
template <typename Time_Traits>
std::size_t epoll_reactor::cancel_timer(
    timer_queue<Time_Traits>& timer_queue, void* token)
{
  mutex::scoped_lock lock(mutex_);
  op_queue<operation> ops;
  std::size_t n = timer_queue.cancel_timer(token, ops);
  lock.unlock();
  io_service_.post_deferred_completions(ops);
  return n;
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_EPOLL)

#endif // ASIO_DETAIL_IMPL_EPOLL_REACTOR_HPP
