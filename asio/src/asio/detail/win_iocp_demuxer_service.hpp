//
// win_iocp_demuxer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_WIN_IOCP_DEMUXER_SERVICE_HPP
#define ASIO_DETAIL_WIN_IOCP_DEMUXER_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#if defined(_WIN32) // This provider is only supported on Win32

#include "asio/basic_demuxer.hpp"
#include "asio/completion_context_locker.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/tss_bool.hpp"

namespace asio {
namespace detail {

class win_iocp_demuxer_service
  : public completion_context_locker
{
public:
  // Constructor.
  win_iocp_demuxer_service(basic_demuxer<win_iocp_demuxer_service>&)
    : iocp_(::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0, 0)),
      outstanding_operations_(0),
      interrupted_(0),
      current_thread_in_pool_()
  {
  }

  // Register a socket with the demuxer.
  void register_socket(socket_type sock)
  {
    HANDLE sock_as_handle = reinterpret_cast<HANDLE>(sock);
    ::CreateIoCompletionPort(sock_as_handle, iocp_.handle, 0, 0);
  }

  // Structure used as the base type for all operations.
  struct operation : public OVERLAPPED
  {
    operation(completion_context& context, bool context_acquired)
      : context_(context),
        context_acquired_(context_acquired)
    {
      ::ZeroMemory(static_cast<OVERLAPPED*>(this), sizeof(OVERLAPPED));
    }

    virtual void do_completion(DWORD last_error, size_t bytes_transferred) = 0;

    completion_context& context_;
    bool context_acquired_;
  };

  // Run the demuxer's event processing loop.
  void run()
  {
    if (::InterlockedExchangeAdd(&outstanding_operations_, 0) == 0)
      return;

    current_thread_in_pool_ = true;

    for (;;)
    {
      // Get the next operation from the queue.
      DWORD bytes_transferred = 0;
      DWORD_PTR completion_key = 0;
      LPOVERLAPPED overlapped = 0;
      ::SetLastError(0);
      ::GetQueuedCompletionStatus(iocp_.handle, &bytes_transferred,
          &completion_key, &overlapped, INFINITE);
      DWORD last_error = ::GetLastError();

      if (overlapped)
      {
        // Dispatch the operation.
        operation* op = static_cast<operation*>(overlapped);
        if (!op->context_acquired_ && !try_acquire(op->context_))
        {
          acquire(op->context_, op);
        }
        else
        {
          op->context_acquired_ = true;
          completion_context& context = op->context_;
          op->do_completion(last_error, bytes_transferred);
          release(context);
          if (::InterlockedDecrement(&outstanding_operations_) == 0)
            interrupt();
        }
      }
      else
      {
        // The interrupted_ flag is always checked to ensure that any leftover
        // interrupts from a previous run invocation are ignored.
        if (::InterlockedExchangeAdd(&interrupted_, 0) != 0)
        {
          // Wake up next thread that is blocked on GetQueuedCompletionStatus.
          ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0);
          break;
        }
      }
    }

    current_thread_in_pool_ = false;
  }

  // Interrupt the demuxer's event processing loop.
  void interrupt()
  {
    if (::InterlockedExchange(&interrupted_, 1) == 0)
      ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0);
  }

  // Reset the demuxer in preparation for a subsequent run invocation.
  void reset()
  {
    ::InterlockedExchange(&interrupted_, 0);
  }

  // Notify the demuxer that an operation has started.
  void operation_started()
  {
    ::InterlockedIncrement(&outstanding_operations_);
  }

  template <typename Handler>
  struct completion_operation : public operation
  {
    completion_operation(completion_context& context, bool context_acquired,
        Handler handler)
      : operation(context, context_acquired),
        handler_(handler)
    {
    }

    virtual void do_completion(DWORD, size_t)
    {
      do_upcall(handler_);
      delete this;
    }

    static void do_upcall(Handler& handler)
    {
      try
      {
        handler();
      }
      catch (...)
      {
      }
    }

    Handler handler_;
  };

  // Notify the demuxer that an operation has completed.
  template <typename Handler>
  void operation_completed(Handler handler, completion_context& context,
      bool allow_nested_delivery)
  {
    if (try_acquire(context))
    {
      if (allow_nested_delivery && current_thread_in_pool_)
      {
        completion_operation<Handler>::do_upcall(handler);
        release(context);
        if (::InterlockedDecrement(&outstanding_operations_) == 0)
          interrupt();
      }
      else
      {
        completion_operation<Handler>* op =
          new completion_operation<Handler>(context, true, handler);
        ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, op);
      }
    }
    else
    {
      completion_operation<Handler>* op =
        new completion_operation<Handler>(context, false, handler);
      acquire(context, op);
    }
  }

  // Notify the demuxer of an operation that started and finished immediately.
  template <typename Handler>
  void operation_immediate(Handler handler, completion_context& context,
      bool allow_nested_delivery)
  {
    operation_started();
    operation_completed(handler, context, allow_nested_delivery);
  }

  // Callback function when a completion context has been acquired.
  void completion_context_acquired(void* arg) throw ()
  {
    operation* op = static_cast<operation*>(arg);
    op->context_acquired_ = true;
    ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, op);
  }

private:
  // The IO completion port used for queueing operations.
  struct iocp_holder
  {
    HANDLE handle;
    iocp_holder(HANDLE h) : handle(h) {}
    ~iocp_holder() { ::CloseHandle(handle); }
  } iocp_;

  // The number of operations that have not yet completed.
  long outstanding_operations_;

  // Flag to indicate whether the event loop has been interrupted.
  long interrupted_;

  // Thread-specific flag to keep track of which threads are in the pool.
  tss_bool current_thread_in_pool_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_DEMUXER_SERVICE_HPP
