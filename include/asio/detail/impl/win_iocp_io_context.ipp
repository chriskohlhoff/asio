//
// detail/impl/win_iocp_io_context.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2025 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_WIN_IOCP_IO_CONTEXT_IPP
#define ASIO_DETAIL_IMPL_WIN_IOCP_IO_CONTEXT_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/config.hpp"
#include "asio/error.hpp"
#include "asio/detail/cstdint.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/limits.hpp"
#include "asio/detail/thread.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/win_iocp_io_context.hpp"

#include "asio/detail/push_options.hpp"

#if defined(ASIO_HAS_IOCP_HIRES_TIMERS)
#include <winternl.h>
#endif // defined(ASIO_HAS_IOCP_HIRES_TIMERS)

namespace asio {
namespace detail {

struct win_iocp_io_context::thread_function
{
  explicit thread_function(win_iocp_io_context* s)
    : this_(s)
  {
  }

  void operator()()
  {
    asio::error_code ec;
    this_->run(ec);
  }

  win_iocp_io_context* this_;
};

struct win_iocp_io_context::work_finished_on_block_exit
{
  ~work_finished_on_block_exit() noexcept(false)
  {
    io_context_->work_finished();
  }

  win_iocp_io_context* io_context_;
};

#if !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
struct win_iocp_io_context::timer_thread_function
{
  void operator()()
  {
    while (::InterlockedExchangeAdd(&io_context_->shutdown_, 0) == 0)
    {
      if (::WaitForSingleObject(io_context_->waitable_timer_.handle,
            INFINITE) == WAIT_OBJECT_0)
      {
        ::InterlockedExchange(&io_context_->dispatch_required_, 1);
        ::PostQueuedCompletionStatus(io_context_->iocp_.handle,
            0, wake_for_dispatch, 0);
      }
    }
  }

  win_iocp_io_context* io_context_;
};
#endif // !defined(ASIO_HAS_IOCP_HIRES_TIMERS)

win_iocp_io_context::win_iocp_io_context(
    asio::execution_context& ctx, bool own_thread)
  : execution_context_service_base<win_iocp_io_context>(ctx),
    iocp_(),
    outstanding_work_(0),
    stopped_(0),
    stop_event_posted_(0),
    shutdown_(0),
    gqcs_timeout_(get_gqcs_timeout()),
    dispatch_required_(0),
    concurrency_hint_(config(ctx).get("scheduler", "concurrency_hint", -1))
{
  ASIO_HANDLER_TRACKING_INIT;

  iocp_.handle = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0,
      static_cast<DWORD>(concurrency_hint_ >= 0
        ? concurrency_hint_ : DWORD(~0)));
  if (!iocp_.handle)
  {
    DWORD last_error = ::GetLastError();
    asio::error_code ec(last_error,
        asio::error::get_system_category());
    asio::detail::throw_error(ec, "iocp");
  }

#if defined(ASIO_HAS_IOCP_HIRES_TIMERS)
  if (FARPROC nt_create_wait_completion_packet_ptr = ::GetProcAddress(
          ::GetModuleHandleA("NTDLL"), "NtCreateWaitCompletionPacket"))
  {
    NtCreateWaitCompletionPacket_ =
        reinterpret_cast<NtCreateWaitCompletionPacket_fn>(
            reinterpret_cast<void*>(nt_create_wait_completion_packet_ptr));
  }
  else
  {
    DWORD last_error = ::GetLastError();
    asio::error_code ec(last_error,
        asio::error::get_system_category());
    asio::detail::throw_error(ec, "timer");
  }

  if (FARPROC nt_associate_wait_completion_packet_ptr = ::GetProcAddress(
          ::GetModuleHandleA("NTDLL"), "NtAssociateWaitCompletionPacket")) {
    NtAssociateWaitCompletionPacket_ =
        reinterpret_cast<NtAssociateWaitCompletionPacket_fn>(
            reinterpret_cast<void*>(nt_associate_wait_completion_packet_ptr));
  }
  else
  {
    DWORD last_error = ::GetLastError();
    asio::error_code ec(last_error,
        asio::error::get_system_category());
    asio::detail::throw_error(ec, "timer");
  }

  if (FARPROC rtl_nt_status_to_dos_error_ptr = ::GetProcAddress(
          ::GetModuleHandleA("NTDLL"), "RtlNtStatusToDosError")) {
    RtlNtStatusToDosError_ =
        reinterpret_cast<RtlNtStatusToDosError_fn>(
            reinterpret_cast<void*>(rtl_nt_status_to_dos_error_ptr));
  }
  else
  {
    DWORD last_error = ::GetLastError();
    asio::error_code ec(last_error,
        asio::error::get_system_category());
    asio::detail::throw_error(ec, "timer");
  }
#endif // defined(ASIO_HAS_IOCP_HIRES_TIMERS)

  if (own_thread)
  {
    ::InterlockedIncrement(&outstanding_work_);
    thread_ = thread(thread_function(this));
  }
}

win_iocp_io_context::win_iocp_io_context(
    win_iocp_io_context::internal, asio::execution_context& ctx)
  : execution_context_service_base<win_iocp_io_context>(ctx),
    iocp_(),
    outstanding_work_(0),
    stopped_(0),
    stop_event_posted_(0),
    shutdown_(0),
    gqcs_timeout_(get_gqcs_timeout()),
    dispatch_required_(0),
    concurrency_hint_(-1)
{
  ASIO_HANDLER_TRACKING_INIT;

  iocp_.handle = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0,
      static_cast<DWORD>(concurrency_hint_ >= 0
        ? concurrency_hint_ : DWORD(~0)));
  if (!iocp_.handle)
  {
    DWORD last_error = ::GetLastError();
    asio::error_code ec(last_error,
        asio::error::get_system_category());
    asio::detail::throw_error(ec, "iocp");
  }
}

win_iocp_io_context::~win_iocp_io_context()
{
  if (thread_.joinable())
  {
    stop();
    thread_.join();
  }
}

void win_iocp_io_context::shutdown()
{
  ::InterlockedExchange(&shutdown_, 1);

#if !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
  if (timer_thread_.joinable())
  {
    LARGE_INTEGER timeout;
    timeout.QuadPart = 1;
    ::SetWaitableTimer(waitable_timer_.handle, &timeout, 1, 0, 0, FALSE);
  }
#endif // !defined(ASIO_HAS_IOCP_HIRES_TIMERS)

  if (thread_.joinable())
  {
    stop();
    thread_.join();
    ::InterlockedDecrement(&outstanding_work_);
  }

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
          &completion_key, &overlapped, gqcs_timeout_);
      if (overlapped)
      {
        ::InterlockedDecrement(&outstanding_work_);
        static_cast<win_iocp_operation*>(overlapped)->destroy();
      }
    }
  }

#if !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
  timer_thread_.join();
#endif // !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
}

asio::error_code win_iocp_io_context::register_handle(
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

size_t win_iocp_io_context::run(asio::error_code& ec)
{
  if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
  {
    stop();
    ec = asio::error_code();
    return 0;
  }

  win_iocp_thread_info this_thread;
  thread_call_stack::context ctx(this, this_thread);

  size_t n = 0;
  while (do_one(INFINITE, this_thread, ec))
    if (n != (std::numeric_limits<size_t>::max)())
      ++n;
  return n;
}

size_t win_iocp_io_context::run_one(asio::error_code& ec)
{
  if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
  {
    stop();
    ec = asio::error_code();
    return 0;
  }

  win_iocp_thread_info this_thread;
  thread_call_stack::context ctx(this, this_thread);

  return do_one(INFINITE, this_thread, ec);
}

size_t win_iocp_io_context::wait_one(long usec, asio::error_code& ec)
{
  if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
  {
    stop();
    ec = asio::error_code();
    return 0;
  }

  win_iocp_thread_info this_thread;
  thread_call_stack::context ctx(this, this_thread);

  return do_one(usec < 0 ? INFINITE : ((usec - 1) / 1000 + 1), this_thread, ec);
}

size_t win_iocp_io_context::poll(asio::error_code& ec)
{
  if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
  {
    stop();
    ec = asio::error_code();
    return 0;
  }

  win_iocp_thread_info this_thread;
  thread_call_stack::context ctx(this, this_thread);

  size_t n = 0;
  while (do_one(0, this_thread, ec))
    if (n != (std::numeric_limits<size_t>::max)())
      ++n;
  return n;
}

size_t win_iocp_io_context::poll_one(asio::error_code& ec)
{
  if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
  {
    stop();
    ec = asio::error_code();
    return 0;
  }

  win_iocp_thread_info this_thread;
  thread_call_stack::context ctx(this, this_thread);

  return do_one(0, this_thread, ec);
}

void win_iocp_io_context::stop()
{
  if (::InterlockedExchange(&stopped_, 1) == 0)
  {
    if (::InterlockedExchange(&stop_event_posted_, 1) == 0)
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
}

bool win_iocp_io_context::can_dispatch()
{
  return thread_call_stack::contains(this) != 0;
}

void win_iocp_io_context::capture_current_exception()
{
  if (thread_info_base* this_thread = thread_call_stack::contains(this))
    this_thread->capture_current_exception();
}

void win_iocp_io_context::post_deferred_completion(win_iocp_operation* op)
{
  // Flag the operation as ready.
  op->ready_ = 1;

  // Enqueue the operation on the I/O completion port.
  if (!::PostQueuedCompletionStatus(iocp_.handle, 0, 0, op))
  {
    // Out of resources. Put on completed queue instead.
    mutex::scoped_lock lock(dispatch_mutex_);
    completed_ops_.push(op);
    ::InterlockedExchange(&dispatch_required_, 1);
  }
}

void win_iocp_io_context::post_deferred_completions(
    op_queue<win_iocp_operation>& ops)
{
  while (win_iocp_operation* op = ops.front())
  {
    ops.pop();

    // Flag the operation as ready.
    op->ready_ = 1;

    // Enqueue the operation on the I/O completion port.
    if (!::PostQueuedCompletionStatus(iocp_.handle, 0, 0, op))
    {
      // Out of resources. Put on completed queue instead.
      mutex::scoped_lock lock(dispatch_mutex_);
      completed_ops_.push(op);
      completed_ops_.push(ops);
      ::InterlockedExchange(&dispatch_required_, 1);
    }
  }
}

void win_iocp_io_context::abandon_operations(
    op_queue<win_iocp_operation>& ops)
{
  while (win_iocp_operation* op = ops.front())
  {
    ops.pop();
    ::InterlockedDecrement(&outstanding_work_);
    op->destroy();
  }
}

void win_iocp_io_context::on_pending(win_iocp_operation* op)
{
  if (::InterlockedCompareExchange(&op->ready_, 1, 0) == 1)
  {
    // Enqueue the operation on the I/O completion port.
    if (!::PostQueuedCompletionStatus(iocp_.handle,
          0, overlapped_contains_result, op))
    {
      // Out of resources. Put on completed queue instead.
      mutex::scoped_lock lock(dispatch_mutex_);
      completed_ops_.push(op);
      ::InterlockedExchange(&dispatch_required_, 1);
    }
  }
}

void win_iocp_io_context::on_completion(win_iocp_operation* op,
    DWORD last_error, DWORD bytes_transferred)
{
  // Flag that the operation is ready for invocation.
  op->ready_ = 1;

  // Store results in the OVERLAPPED structure.
  op->Internal = reinterpret_cast<ulong_ptr_t>(
      &asio::error::get_system_category());
  op->Offset = last_error;
  op->OffsetHigh = bytes_transferred;

  // Enqueue the operation on the I/O completion port.
  if (!::PostQueuedCompletionStatus(iocp_.handle,
        0, overlapped_contains_result, op))
  {
    // Out of resources. Put on completed queue instead.
    mutex::scoped_lock lock(dispatch_mutex_);
    completed_ops_.push(op);
    ::InterlockedExchange(&dispatch_required_, 1);
  }
}

void win_iocp_io_context::on_completion(win_iocp_operation* op,
    const asio::error_code& ec, DWORD bytes_transferred)
{
  // Flag that the operation is ready for invocation.
  op->ready_ = 1;

  // Store results in the OVERLAPPED structure.
  op->Internal = reinterpret_cast<ulong_ptr_t>(&ec.category());
  op->Offset = ec.value();
  op->OffsetHigh = bytes_transferred;

  // Enqueue the operation on the I/O completion port.
  if (!::PostQueuedCompletionStatus(iocp_.handle,
        0, overlapped_contains_result, op))
  {
    // Out of resources. Put on completed queue instead.
    mutex::scoped_lock lock(dispatch_mutex_);
    completed_ops_.push(op);
    ::InterlockedExchange(&dispatch_required_, 1);
  }
}

size_t win_iocp_io_context::do_one(DWORD msec,
    win_iocp_thread_info& this_thread, asio::error_code& ec)
{
  for (;;)
  {
    // Try to acquire responsibility for dispatching timers and completed ops.
    if (::InterlockedCompareExchange(&dispatch_required_, 0, 1) == 1)
    {
      mutex::scoped_lock lock(dispatch_mutex_);

      // Dispatch pending timers and operations.
      op_queue<win_iocp_operation> ops;
      ops.push(completed_ops_);
      timer_queues_.get_ready_timers(ops);
      post_deferred_completions(ops);
      update_timeout();
    }

    // Get the next operation from the queue.
    DWORD bytes_transferred = 0;
    dword_ptr_t completion_key = 0;
    LPOVERLAPPED overlapped = 0;
    ::SetLastError(0);
    BOOL ok = ::GetQueuedCompletionStatus(iocp_.handle,
        &bytes_transferred, &completion_key, &overlapped,
        msec < gqcs_timeout_ ? msec : gqcs_timeout_);
    DWORD last_error = ::GetLastError();

    if (overlapped)
    {
      win_iocp_operation* op = static_cast<win_iocp_operation*>(overlapped);
      asio::error_code result_ec(last_error,
          asio::error::get_system_category());

      // We may have been passed the last_error and bytes_transferred in the
      // OVERLAPPED structure itself.
      if (completion_key == overlapped_contains_result)
      {
        result_ec = asio::error_code(static_cast<int>(op->Offset),
            *reinterpret_cast<asio::error_category*>(op->Internal));
        bytes_transferred = op->OffsetHigh;
      }

      // Otherwise ensure any result has been saved into the OVERLAPPED
      // structure.
      else
      {
        op->Internal = reinterpret_cast<ulong_ptr_t>(&result_ec.category());
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

        op->complete(this, result_ec, bytes_transferred);
        this_thread.rethrow_pending_exception();
        ec = asio::error_code();
        return 1;
      }
    }
    else if (!ok)
    {
      if (last_error != WAIT_TIMEOUT)
      {
        ec = asio::error_code(last_error,
            asio::error::get_system_category());
        return 0;
      }

      // If we're waiting indefinitely we need to keep going until we get a
      // real handler.
      if (msec == INFINITE)
        continue;

      ec = asio::error_code();
      return 0;
    }
    else if (completion_key == wake_for_dispatch)
    {
      // We have been woken up to try to acquire responsibility for dispatching
      // timers and completed operations.
#if defined(ASIO_HAS_IOCP_HIRES_TIMERS)
      mutex::scoped_lock lock(dispatch_mutex_);
      ::InterlockedExchange(&dispatch_required_, 1);
#endif // defined(ASIO_HAS_IOCP_HIRES_TIMERS)
    }
    else
    {
      // Indicate that there is no longer an in-flight stop event.
      ::InterlockedExchange(&stop_event_posted_, 0);

      // The stopped_ flag is always checked to ensure that any leftover
      // stop events from a previous run invocation are ignored.
      if (::InterlockedExchangeAdd(&stopped_, 0) != 0)
      {
        // Wake up next thread that is blocked on GetQueuedCompletionStatus.
        if (::InterlockedExchange(&stop_event_posted_, 1) == 0)
        {
          if (!::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0))
          {
            last_error = ::GetLastError();
            ec = asio::error_code(last_error,
                asio::error::get_system_category());
            return 0;
          }
        }

        ec = asio::error_code();
        return 0;
      }
    }
  }
}

DWORD win_iocp_io_context::get_gqcs_timeout()
{
#if !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0600)
  OSVERSIONINFOEX osvi;
  ZeroMemory(&osvi, sizeof(osvi));
  osvi.dwOSVersionInfoSize = sizeof(osvi);
  osvi.dwMajorVersion = 6ul;

  const uint64_t condition_mask = ::VerSetConditionMask(
      0, VER_MAJORVERSION, VER_GREATER_EQUAL);

  if (!!::VerifyVersionInfo(&osvi, VER_MAJORVERSION, condition_mask))
    return INFINITE;

  return default_gqcs_timeout;
#else // !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0600)
  return INFINITE;
#endif // !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0600)
}

void win_iocp_io_context::do_add_timer_queue(timer_queue_base& queue)
{
  mutex::scoped_lock lock(dispatch_mutex_);

  timer_queues_.insert(&queue);

#if defined(ASIO_HAS_IOCP_HIRES_TIMERS)
  if (!iocp_wait_handle_.handle)
  {
    NTSTATUS status = NtCreateWaitCompletionPacket_(&iocp_wait_handle_.handle, GENERIC_ALL, 0);
    if (!NT_SUCCESS(status) || (iocp_wait_handle_.handle == 0)) {
      DWORD win32_error = RtlNtStatusToDosError_(status);
      asio::error_code ec(win32_error, asio::error::get_system_category());
      asio::detail::throw_error(ec, "timer");
    }
  }
#endif // defined(ASIO_HAS_IOCP_HIRES_TIMERS)

  if (!waitable_timer_.handle)
  {
#if defined(ASIO_HAS_IOCP_HIRES_TIMERS)
    waitable_timer_.handle = ::CreateWaitableTimerExW(
          0, 0, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, SYNCHRONIZE | TIMER_MODIFY_STATE);
#else
    waitable_timer_.handle = ::CreateWaitableTimer(0, FALSE, 0);
#endif // defined(ASIO_HAS_IOCP_HIRES_TIMERS)
    if (waitable_timer_.handle == 0)
    {
      DWORD last_error = ::GetLastError();
      asio::error_code ec(last_error,
          asio::error::get_system_category());
      asio::detail::throw_error(ec, "timer");
    }

    LARGE_INTEGER timeout;
    timeout.QuadPart = -max_timeout_usec;
    timeout.QuadPart *= 10;
    ::SetWaitableTimer(waitable_timer_.handle,
        &timeout, max_timeout_msec, 0, 0, FALSE);
#if defined(ASIO_HAS_IOCP_HIRES_TIMERS)
    NTSTATUS status = NtAssociateWaitCompletionPacket_(iocp_wait_handle_.handle, iocp_.handle,
                                                       waitable_timer_.handle, (PVOID)wake_for_dispatch,
                                                       0, 0, 0, 0);
    if (!NT_SUCCESS(status)) {
      DWORD win32_error = RtlNtStatusToDosError_(status);
      asio::error_code ec(win32_error, asio::error::get_system_category());
      asio::detail::throw_error(ec, "timer");
    }
#endif // defined(ASIO_HAS_IOCP_HIRES_TIMERS)
  }

#if !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
  if (!timer_thread_.joinable())
  {
    timer_thread_function thread_function = { this };
    timer_thread_ = thread(thread_function, 65536);
  }
#endif // !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
}

void win_iocp_io_context::do_remove_timer_queue(timer_queue_base& queue)
{
  mutex::scoped_lock lock(dispatch_mutex_);

  timer_queues_.erase(&queue);
}

void win_iocp_io_context::update_timeout()
{

#if !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
  if (timer_thread_.joinable())
  {
#endif // !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
    // There's no point updating the waitable timer if the new timeout period
    // exceeds the maximum timeout. In that case, we might as well wait for the
    // existing period of the timer to expire.
    long timeout_usec = timer_queues_.wait_duration_usec(max_timeout_usec);
    if (timeout_usec < max_timeout_usec)
    {
      LARGE_INTEGER timeout;
      timeout.QuadPart = -timeout_usec;
      timeout.QuadPart *= 10;
      ::SetWaitableTimer(waitable_timer_.handle,
          &timeout, max_timeout_msec, 0, 0, FALSE);
#if defined(ASIO_HAS_IOCP_HIRES_TIMERS)
      NtAssociateWaitCompletionPacket_(iocp_wait_handle_.handle, iocp_.handle,
                                       waitable_timer_.handle,
                                       (PVOID)wake_for_dispatch, 0, 0, 0, 0);
#endif // defined(ASIO_HAS_IOCP_HIRES_TIMERS)
    }
#if !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
  }
#endif // !defined(ASIO_HAS_IOCP_HIRES_TIMERS)
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_IMPL_WIN_IOCP_IO_CONTEXT_IPP
