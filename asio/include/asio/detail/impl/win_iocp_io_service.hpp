//
// detail/impl/win_iocp_io_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_WIN_IOCP_IO_SERVICE_HPP
#define ASIO_DETAIL_IMPL_WIN_IOCP_IO_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/call_stack.hpp"
#include "asio/detail/completion_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Handler>
void win_iocp_io_service::dispatch(Handler handler)
{
  if (call_stack<win_iocp_io_service>::contains(this))
  {
    asio::detail::fenced_block b;
    asio_handler_invoke_helpers::invoke(handler, handler);
  }
  else
    post(handler);
}

template <typename Handler>
void win_iocp_io_service::post(Handler handler)
{
  // Allocate and construct an operation to wrap the handler.
  typedef completion_handler<Handler> op;
  typename op::ptr p = { boost::addressof(handler),
    asio_handler_alloc_helpers::allocate(
      sizeof(op), handler), 0 };
  p.p = new (p.v) op(handler);

  post_immediate_completion(p.p);
  p.v = p.p = 0;
}

template <typename Time_Traits>
void win_iocp_io_service::add_timer_queue(
    timer_queue<Time_Traits>& timer_queue)
{
  asio::detail::mutex::scoped_lock lock(timer_mutex_);
  timer_queues_.insert(&timer_queue);
}

template <typename Time_Traits>
void win_iocp_io_service::remove_timer_queue(
    timer_queue<Time_Traits>& timer_queue)
{
  asio::detail::mutex::scoped_lock lock(timer_mutex_);
  timer_queues_.erase(&timer_queue);
}

template <typename Time_Traits>
void win_iocp_io_service::schedule_timer(timer_queue<Time_Traits>& timer_queue,
    const typename Time_Traits::time_type& time, timer_op* op, void* token)
{
  // If the service has been shut down we silently discard the timer.
  if (::InterlockedExchangeAdd(&shutdown_, 0) != 0)
    return;

  asio::detail::mutex::scoped_lock lock(timer_mutex_);
  bool interrupt = timer_queue.enqueue_timer(time, op, token);
  work_started();
  if (interrupt && !timer_interrupt_issued_)
  {
    timer_interrupt_issued_ = true;
    lock.unlock();
    ::PostQueuedCompletionStatus(iocp_.handle,
        0, steal_timer_dispatching, 0);
  }
}

template <typename Time_Traits>
std::size_t win_iocp_io_service::cancel_timer(
    timer_queue<Time_Traits>& timer_queue, void* token)
{
  // If the service has been shut down we silently ignore the cancellation.
  if (::InterlockedExchangeAdd(&shutdown_, 0) != 0)
    return 0;

  asio::detail::mutex::scoped_lock lock(timer_mutex_);
  op_queue<win_iocp_operation> ops;
  std::size_t n = timer_queue.cancel_timer(token, ops);
  post_deferred_completions(ops);
  if (n > 0 && !timer_interrupt_issued_)
  {
    timer_interrupt_issued_ = true;
    lock.unlock();
    ::PostQueuedCompletionStatus(iocp_.handle,
        0, steal_timer_dispatching, 0);
  }
  return n;
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_IMPL_WIN_IOCP_IO_SERVICE_HPP
