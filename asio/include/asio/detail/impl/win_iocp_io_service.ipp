//
// detail/impl/win_iocp_io_service.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_WIN_IOCP_IO_SERVICE_IPP
#define ASIO_DETAIL_IMPL_WIN_IOCP_IO_SERVICE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include <boost/limits.hpp>
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/win_iocp_io_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct win_iocp_io_service::work_finished_on_block_exit
{
  ~work_finished_on_block_exit()
  {
    io_service_->work_finished();
  }

  win_iocp_io_service* io_service_;
};

win_iocp_io_service::win_iocp_io_service(asio::io_service& io_service)
  : asio::detail::service_base<win_iocp_io_service>(io_service),
    iocp_(),
    outstanding_work_(0),
    stopped_(0),
    shutdown_(0),
    timer_thread_(0),
    timer_interrupt_issued_(false)
{
}

void win_iocp_io_service::init(size_t concurrency_hint)
{
  iocp_.handle = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0,
      static_cast<DWORD>((std::min<size_t>)(concurrency_hint, DWORD(~0))));
  if (!iocp_.handle)
  {
    DWORD last_error = ::GetLastError();
    asio::error_code ec(last_error,
        asio::error::get_system_category());
    asio::detail::throw_error(ec, "iocp");
  }
}

void win_iocp_io_service::shutdown_service()
{
  ::InterlockedExchange(&shutdown_, 1);

  while (::InterlockedExchangeAdd(&outstanding_work_, 0) > 0)
  {
    op_queue<win_iocp_operation> ops;
    timer_queues_.get_all_timers(ops);
    ops.push(completed_ops_);
    if (!ops.empty())
    {
      while (win_iocp_operation* op = ops.front())
      {
        ops.pop();
        ::InterlockedDecrement(&outstanding_work_);
        op->destroy();
      }
    }
    else
    {
      DWORD bytes_transferred = 0;
      dword_ptr_t completion_key = 0;
      LPOVERLAPPED overlapped = 0;
      ::GetQueuedCompletionStatus(iocp_.handle, &bytes_transferred,
          &completion_key, &overlapped, max_timeout);
      if (overlapped)
      {
        ::InterlockedDecrement(&outstanding_work_);
        static_cast<win_iocp_operation*>(overlapped)->destroy();
      }
    }
  }
}

asio::error_code win_iocp_io_service::register_handle(
    HANDLE handle, asio::error_code& ec)
{
  if (::CreateIoCompletionPort(handle, iocp_.handle, 0, 0) == 0)
  {
    DWORD last_error = ::GetLastError();
    ec = asio::error_code(last_error,
        asio::error::get_system_category());
  }
  else
  {
    ec = asio::error_code();
  }
  return ec;
}

size_t win_iocp_io_service::run(asio::error_code& ec)
{
  if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
  {
    stop();
    ec = asio::error_code();
    return 0;
  }

  call_stack<win_iocp_io_service>::context ctx(this);

  size_t n = 0;
  while (do_one(true, ec))
    if (n != (std::numeric_limits<size_t>::max)())
      ++n;
  return n;
}

size_t win_iocp_io_service::run_one(asio::error_code& ec)
{
  if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
  {
    stop();
    ec = asio::error_code();
    return 0;
  }

  call_stack<win_iocp_io_service>::context ctx(this);

  return do_one(true, ec);
}

size_t win_iocp_io_service::poll(asio::error_code& ec)
{
  if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
  {
    stop();
    ec = asio::error_code();
    return 0;
  }

  call_stack<win_iocp_io_service>::context ctx(this);

  size_t n = 0;
  while (do_one(false, ec))
    if (n != (std::numeric_limits<size_t>::max)())
      ++n;
  return n;
}

size_t win_iocp_io_service::poll_one(asio::error_code& ec)
{
  if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
  {
    stop();
    ec = asio::error_code();
    return 0;
  }

  call_stack<win_iocp_io_service>::context ctx(this);

  return do_one(false, ec);
}

void win_iocp_io_service::stop()
{
  if (::InterlockedExchange(&stopped_, 1) == 0)
  {
    if (!::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0))
    {
      DWORD last_error = ::GetLastError();
      asio::error_code ec(last_error,
          asio::error::get_system_category());
      asio::detail::throw_error(ec, "pqcs");
    }
  }
}

void win_iocp_io_service::post_deferred_completion(win_iocp_operation* op)
{
  // Flag the operation as ready.
  op->ready_ = 1;

  // Enqueue the operation on the I/O completion port.
  if (!::PostQueuedCompletionStatus(iocp_.handle,
        0, overlapped_contains_result, op))
  {
    // Out of resources. Put on completed queue instead.
    asio::detail::mutex::scoped_lock lock(timer_mutex_);
    completed_ops_.push(op);
  }
}

void win_iocp_io_service::post_deferred_completions(
    op_queue<win_iocp_operation>& ops)
{
  while (win_iocp_operation* op = ops.front())
  {
    ops.pop();

    // Flag the operation as ready.
    op->ready_ = 1;

    // Enqueue the operation on the I/O completion port.
    if (!::PostQueuedCompletionStatus(iocp_.handle,
          0, overlapped_contains_result, op))
    {
      // Out of resources. Put on completed queue instead.
      asio::detail::mutex::scoped_lock lock(timer_mutex_);
      completed_ops_.push(op);
      completed_ops_.push(ops);
    }
  }
}

void win_iocp_io_service::on_pending(win_iocp_operation* op)
{
  if (::InterlockedCompareExchange(&op->ready_, 1, 0) == 1)
  {
    // Enqueue the operation on the I/O completion port.
    if (!::PostQueuedCompletionStatus(iocp_.handle,
          0, overlapped_contains_result, op))
    {
      // Out of resources. Put on completed queue instead.
      asio::detail::mutex::scoped_lock lock(timer_mutex_);
      completed_ops_.push(op);
    }
  }
}

void win_iocp_io_service::on_completion(win_iocp_operation* op,
    DWORD last_error, DWORD bytes_transferred)
{
  // Flag that the operation is ready for invocation.
  op->ready_ = 1;

  // Store results in the OVERLAPPED structure.
  op->Internal = asio::error::get_system_category();
  op->Offset = last_error;
  op->OffsetHigh = bytes_transferred;

  // Enqueue the operation on the I/O completion port.
  if (!::PostQueuedCompletionStatus(iocp_.handle,
        0, overlapped_contains_result, op))
  {
    // Out of resources. Put on completed queue instead.
    asio::detail::mutex::scoped_lock lock(timer_mutex_);
    completed_ops_.push(op);
  }
}

void win_iocp_io_service::on_completion(win_iocp_operation* op,
    const asio::error_code& ec, DWORD bytes_transferred)
{
  // Flag that the operation is ready for invocation.
  op->ready_ = 1;

  // Store results in the OVERLAPPED structure.
  op->Internal = ec.category();
  op->Offset = ec.value();
  op->OffsetHigh = bytes_transferred;

  // Enqueue the operation on the I/O completion port.
  if (!::PostQueuedCompletionStatus(iocp_.handle,
        0, overlapped_contains_result, op))
  {
    // Out of resources. Put on completed queue instead.
    asio::detail::mutex::scoped_lock lock(timer_mutex_);
    completed_ops_.push(op);
  }
}

size_t win_iocp_io_service::do_one(bool block, asio::error_code& ec)
{
  long this_thread_id = static_cast<long>(::GetCurrentThreadId());

  for (;;)
  {
    // Try to acquire responsibility for dispatching timers.
    bool dispatching_timers = (::InterlockedCompareExchange(
          &timer_thread_, this_thread_id, 0) == 0);

    // Calculate timeout for GetQueuedCompletionStatus call.
    DWORD timeout = max_timeout;
    if (dispatching_timers)
    {
      asio::detail::mutex::scoped_lock lock(timer_mutex_);
      timer_interrupt_issued_ = false;
      timeout = timer_queues_.wait_duration_msec(max_timeout);
    }

    // Get the next operation from the queue.
    DWORD bytes_transferred = 0;
    dword_ptr_t completion_key = 0;
    LPOVERLAPPED overlapped = 0;
    ::SetLastError(0);
    BOOL ok = ::GetQueuedCompletionStatus(iocp_.handle, &bytes_transferred,
        &completion_key, &overlapped, block ? timeout : 0);
    DWORD last_error = ::GetLastError();

    // Dispatch any pending timers.
    if (dispatching_timers)
    {
      asio::detail::mutex::scoped_lock lock(timer_mutex_);
      op_queue<win_iocp_operation> ops;
      ops.push(completed_ops_);
      timer_queues_.get_ready_timers(ops);
      post_deferred_completions(ops);
    }

    if (!ok && overlapped == 0)
    {
      if (block && last_error == WAIT_TIMEOUT)
      {
        // Relinquish responsibility for dispatching timers.
        if (dispatching_timers)
        {
          ::InterlockedCompareExchange(&timer_thread_, 0, this_thread_id);
        }

        continue;
      }

      // Transfer responsibility for dispatching timers to another thread.
      if (dispatching_timers && ::InterlockedCompareExchange(
            &timer_thread_, 0, this_thread_id) == this_thread_id)
      {
        ::PostQueuedCompletionStatus(iocp_.handle,
            0, transfer_timer_dispatching, 0);
      }

      ec = asio::error_code();
      return 0;
    }
    else if (overlapped)
    {
      win_iocp_operation* op = static_cast<win_iocp_operation*>(overlapped);
      asio::error_code result_ec(last_error,
          asio::error::get_system_category());

      // Transfer responsibility for dispatching timers to another thread.
      if (dispatching_timers && ::InterlockedCompareExchange(
            &timer_thread_, 0, this_thread_id) == this_thread_id)
      {
        ::PostQueuedCompletionStatus(iocp_.handle,
            0, transfer_timer_dispatching, 0);
      }

      // We may have been passed the last_error and bytes_transferred in the
      // OVERLAPPED structure itself.
      if (completion_key == overlapped_contains_result)
      {
        result_ec = asio::error_code(static_cast<int>(op->Offset),
            static_cast<asio::error_category>(op->Internal));
        bytes_transferred = op->OffsetHigh;
      }

      // Otherwise ensure any result has been saved into the OVERLAPPED
      // structure.
      else
      {
        op->Internal = result_ec.category();
        op->Offset = result_ec.value();
        op->OffsetHigh = bytes_transferred;
      }

      // Dispatch the operation only if ready. The operation may not be ready
      // if the initiating function (e.g. a call to WSARecv) has not yet
      // returned. This is because the initiating function still wants access
      // to the operation's OVERLAPPED structure.
      if (::InterlockedCompareExchange(&op->ready_, 1, 0) == 1)
      {
        // Ensure the count of outstanding work is decremented on block exit.
        work_finished_on_block_exit on_exit = { this };
        (void)on_exit;

        op->complete(*this, result_ec, bytes_transferred);
        ec = asio::error_code();
        return 1;
      }
    }
    else if (completion_key == transfer_timer_dispatching)
    {
      // Woken up to try to acquire responsibility for dispatching timers.
      ::InterlockedCompareExchange(&timer_thread_, 0, this_thread_id);
    }
    else if (completion_key == steal_timer_dispatching)
    {
      // Woken up to steal responsibility for dispatching timers.
      ::InterlockedExchange(&timer_thread_, 0);
    }
    else
    {
      // Relinquish responsibility for dispatching timers. If the io_service
      // is not being stopped then the thread will get an opportunity to
      // reacquire timer responsibility on the next loop iteration.
      if (dispatching_timers)
      {
        ::InterlockedCompareExchange(&timer_thread_, 0, this_thread_id);
      }

      // The stopped_ flag is always checked to ensure that any leftover
      // interrupts from a previous run invocation are ignored.
      if (::InterlockedExchangeAdd(&stopped_, 0) != 0)
      {
        // Wake up next thread that is blocked on GetQueuedCompletionStatus.
        if (!::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0))
        {
          last_error = ::GetLastError();
          ec = asio::error_code(last_error,
              asio::error::get_system_category());
          return 0;
        }

        ec = asio::error_code();
        return 0;
      }
    }
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_IMPL_WIN_IOCP_IO_SERVICE_IPP
