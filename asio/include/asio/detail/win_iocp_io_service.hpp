//
// win_iocp_io_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_IO_SERVICE_HPP
#define ASIO_DETAIL_WIN_IOCP_IO_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/win_iocp_io_service_fwd.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/push_options.hpp"
#include <boost/limits.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/io_service.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/call_stack.hpp"
#include "asio/detail/completion_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/service_base.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/timer_op.hpp"
#include "asio/detail/timer_queue_base.hpp"
#include "asio/detail/timer_queue_fwd.hpp"
#include "asio/detail/timer_queue_set.hpp"
#include "asio/detail/win_iocp_operation.hpp"

namespace asio {
namespace detail {

class timer_op;

class win_iocp_io_service
  : public asio::detail::service_base<win_iocp_io_service>
{
public:
  typedef win_iocp_operation operation;

  // Constructor.
  win_iocp_io_service(asio::io_service& io_service)
    : asio::detail::service_base<win_iocp_io_service>(io_service),
      iocp_(),
      outstanding_work_(0),
      stopped_(0),
      shutdown_(0),
      timer_thread_(0),
      timer_interrupt_issued_(false)
  {
  }

  void init(size_t concurrency_hint)
  {
    iocp_.handle = ::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0,
        static_cast<DWORD>((std::min<size_t>)(concurrency_hint, DWORD(~0))));
    if (!iocp_.handle)
    {
      DWORD last_error = ::GetLastError();
      asio::system_error e(
          asio::error_code(last_error,
            asio::error::get_system_category()),
          "iocp");
      boost::throw_exception(e);
    }
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    ::InterlockedExchange(&shutdown_, 1);

    while (::InterlockedExchangeAdd(&outstanding_work_, 0) > 0)
    {
      op_queue<operation> ops;
      timer_queues_.get_all_timers(ops);
      ops.push(completed_ops_);
      if (!ops.empty())
      {
        while (operation* op = ops.front())
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
          static_cast<operation*>(overlapped)->destroy();
        }
      }
    }
  }

  // Initialise the task. Nothing to do here.
  void init_task()
  {
  }

  // Register a handle with the IO completion port.
  asio::error_code register_handle(
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

  // Run the event loop until stopped or no more work.
  size_t run(asio::error_code& ec)
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

  // Run until stopped or one operation is performed.
  size_t run_one(asio::error_code& ec)
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

  // Poll for operations without blocking.
  size_t poll(asio::error_code& ec)
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

  // Poll for one operation without blocking.
  size_t poll_one(asio::error_code& ec)
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

  // Stop the event processing loop.
  void stop()
  {
    if (::InterlockedExchange(&stopped_, 1) == 0)
    {
      if (!::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0))
      {
        DWORD last_error = ::GetLastError();
        asio::system_error e(
            asio::error_code(last_error,
              asio::error::get_system_category()),
            "pqcs");
        boost::throw_exception(e);
      }
    }
  }

  // Reset in preparation for a subsequent run invocation.
  void reset()
  {
    ::InterlockedExchange(&stopped_, 0);
  }

  // Notify that some work has started.
  void work_started()
  {
    ::InterlockedIncrement(&outstanding_work_);
  }

  // Notify that some work has finished.
  void work_finished()
  {
    if (::InterlockedDecrement(&outstanding_work_) == 0)
      stop();
  }

  // Request invocation of the given handler.
  template <typename Handler>
  void dispatch(Handler handler)
  {
    if (call_stack<win_iocp_io_service>::contains(this))
    {
      asio::detail::fenced_block b;
      asio_handler_invoke_helpers::invoke(handler, handler);
    }
    else
      post(handler);
  }

  // Request invocation of the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef completion_handler<Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, handler);

    post_immediate_completion(ptr.get());
    ptr.release();
  }

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() has not yet been called for the operation.
  void post_immediate_completion(operation* op)
  {
    work_started();
    post_deferred_completion(op);
  }

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() was previously called for the operation.
  void post_deferred_completion(operation* op)
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

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() was previously called for the operations.
  void post_deferred_completions(op_queue<operation>& ops)
  {
    while (operation* op = ops.front())
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

  // Called after starting an overlapped I/O operation that did not complete
  // immediately. The caller must have already called work_started() prior to
  // starting the operation.
  void on_pending(operation* op)
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

  // Called after starting an overlapped I/O operation that completed
  // immediately. The caller must have already called work_started() prior to
  // starting the operation.
  void on_completion(operation* op,
      DWORD last_error = 0, DWORD bytes_transferred = 0)
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

  // Called after starting an overlapped I/O operation that completed
  // immediately. The caller must have already called work_started() prior to
  // starting the operation.
  void on_completion(operation* op,
      const asio::error_code& ec, DWORD bytes_transferred = 0)
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

  // Add a new timer queue to the service.
  template <typename Time_Traits>
  void add_timer_queue(timer_queue<Time_Traits>& timer_queue)
  {
    asio::detail::mutex::scoped_lock lock(timer_mutex_);
    timer_queues_.insert(&timer_queue);
  }

  // Remove a timer queue from the service.
  template <typename Time_Traits>
  void remove_timer_queue(timer_queue<Time_Traits>& timer_queue)
  {
    asio::detail::mutex::scoped_lock lock(timer_mutex_);
    timer_queues_.erase(&timer_queue);
  }

  // Schedule a new operation in the given timer queue to expire at the
  // specified absolute time.
  template <typename Time_Traits>
  void schedule_timer(timer_queue<Time_Traits>& timer_queue,
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

  // Cancel the timer associated with the given token. Returns the number of
  // handlers that have been posted or dispatched.
  template <typename Time_Traits>
  std::size_t cancel_timer(timer_queue<Time_Traits>& timer_queue, void* token)
  {
    // If the service has been shut down we silently ignore the cancellation.
    if (::InterlockedExchangeAdd(&shutdown_, 0) != 0)
      return 0;

    asio::detail::mutex::scoped_lock lock(timer_mutex_);
    op_queue<operation> ops;
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

private:
#if defined(WINVER) && (WINVER < 0x0500)
  typedef DWORD dword_ptr_t;
  typedef ULONG ulong_ptr_t;
#else // defined(WINVER) && (WINVER < 0x0500)
  typedef DWORD_PTR dword_ptr_t;
  typedef ULONG_PTR ulong_ptr_t;
#endif // defined(WINVER) && (WINVER < 0x0500)

  // Dequeues at most one operation from the I/O completion port, and then
  // executes it. Returns the number of operations that were dequeued (i.e.
  // either 0 or 1).
  size_t do_one(bool block, asio::error_code& ec)
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
        timeout = get_timeout();
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
        op_queue<operation> ops;
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
        operation* op = static_cast<operation*>(overlapped);
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

  // Get the timeout value for the GetQueuedCompletionStatus call. The timeout
  // value is returned as a number of milliseconds. We will wait no longer than
  // 1000 milliseconds.
  DWORD get_timeout()
  {
    return timer_queues_.wait_duration_msec(max_timeout);
  }

  // Helper class to call work_finished() on block exit.
  struct work_finished_on_block_exit
  {
    ~work_finished_on_block_exit()
    {
      io_service_->work_finished();
    }

    win_iocp_io_service* io_service_;
  };

  // The IO completion port used for queueing operations.
  struct iocp_holder
  {
    HANDLE handle;
    iocp_holder() : handle(0) {}
    ~iocp_holder() { if (handle) ::CloseHandle(handle); }
  } iocp_;

  // The count of unfinished work.
  long outstanding_work_;

  // Flag to indicate whether the event loop has been stopped.
  long stopped_;

  // Flag to indicate whether the service has been shut down.
  long shutdown_;

  enum
  {
    // Maximum GetQueuedCompletionStatus timeout, in milliseconds.
    max_timeout = 500,

    // Completion key value to indicate that responsibility for dispatching
    // timers is being cooperatively transferred from one thread to another.
    transfer_timer_dispatching = 1,

    // Completion key value to indicate that responsibility for dispatching
    // timers should be stolen from another thread.
    steal_timer_dispatching = 2,

    // Completion key value to indicate that an operation has posted with the
    // original last_error and bytes_transferred values stored in the fields of
    // the OVERLAPPED structure.
    overlapped_contains_result = 3
  };

  // The thread that's currently in charge of dispatching timers.
  long timer_thread_;

  // Mutex for protecting access to the timer queues.
  mutex timer_mutex_;

  // Whether a thread has been interrupted to process a new timeout.
  bool timer_interrupt_issued_;

  // The timer queues.
  timer_queue_set timer_queues_;

  // The operations that are ready to dispatch.
  op_queue<operation> completed_ops_;
};

} // namespace detail
} // namespace asio

#endif // defined(ASIO_HAS_IOCP)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_IO_SERVICE_HPP
