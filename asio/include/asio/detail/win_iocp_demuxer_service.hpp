//
// win_iocp_demuxer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
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
#include "asio/detail/socket_types.hpp"
#include "asio/detail/tss_bool.hpp"

namespace asio {
namespace detail {

class win_iocp_demuxer_service
{
public:
  // The demuxer type for this service.
  typedef basic_demuxer<win_iocp_demuxer_service> demuxer_type;

  // Constructor.
  win_iocp_demuxer_service(demuxer_type& demuxer)
    : demuxer_(demuxer),
      iocp_(::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0, 0)),
      outstanding_operations_(0),
      interrupted_(0),
      current_thread_in_pool_()
  {
  }

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Register a socket with the demuxer.
  void register_socket(socket_type sock)
  {
    HANDLE sock_as_handle = reinterpret_cast<HANDLE>(sock);
    ::CreateIoCompletionPort(sock_as_handle, iocp_.handle, 0, 0);
  }

  // Structure used as the base type for all operations.
  struct operation
    : public OVERLAPPED
  {
    operation(bool context_acquired)
      : context_acquired_(context_acquired)
    {
      ::ZeroMemory(static_cast<OVERLAPPED*>(this), sizeof(OVERLAPPED));
    }

    virtual ~operation()
    {
    }

    // Run the completion. Returns true if the operation is complete.
    virtual bool do_completion(HANDLE iocp, DWORD last_error,
        size_t bytes_transferred) = 0;

    // Ensure that the context has been acquired. Returns true if the context
    // was acquired and the operation can proceed immediately, false otherwise.
    template <typename Completion_Context>
    bool acquire_context(HANDLE iocp, Completion_Context context)
    {
      if (context_acquired_)
        return true;

      if (context.try_acquire())
      {
        context_acquired_ = true;
        return true;
      }

      context.acquire(bind_handler(&operation::context_acquired, iocp, this));
      return false;
    }

    // Ensure that the context has been released.
    template <typename Completion_Context>
    void release_context(Completion_Context context)
    {
      if (context_acquired_)
      {
        context_acquired_ = false;
        context.release();
      }
    }

    static void context_acquired(HANDLE iocp, operation* op)
    {
      op->context_acquired_ = true;
      ::PostQueuedCompletionStatus(iocp, 0, 0, op);
    }

  private:
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
        if (op->do_completion(iocp_.handle, last_error, bytes_transferred))
          if (::InterlockedDecrement(&outstanding_operations_) == 0)
            interrupt();
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

  template <typename Handler, typename Completion_Context>
  struct completion_operation
    : public operation
  {
    completion_operation(Handler handler, Completion_Context context,
        bool context_acquired)
      : operation(context_acquired),
        handler_(handler),
        context_(context)
    {
    }

    virtual bool do_completion(HANDLE iocp, DWORD, size_t)
    {
      if (!acquire_context(iocp, context_))
        return false;

      do_upcall(handler_);
      context_.release();
      delete this;
      return true;
    }

    static void do_upcall(Handler handler)
    {
      try
      {
        handler();
      }
      catch (...)
      {
      }
    }

  private:
    Handler handler_;
    Completion_Context context_;
  };

  // Notify the demuxer that an operation has completed.
  template <typename Handler, typename Completion_Context>
  void operation_completed(Handler handler, Completion_Context context,
      bool allow_nested_delivery)
  {
    if (context.try_acquire())
    {
      if (allow_nested_delivery && current_thread_in_pool_)
      {
        completion_operation<Handler, Completion_Context>::do_upcall(handler);
        context.release();
        if (::InterlockedDecrement(&outstanding_operations_) == 0)
          interrupt();
      }
      else
      {
        operation* op = new completion_operation<Handler, Completion_Context>(
            handler, context, true);
        ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, op);
      }
    }
    else
    {
      operation* op = new completion_operation<Handler, Completion_Context>(
          handler, context, false);
      context.acquire(
          bind_handler(&operation::context_acquired, iocp_.handle, op));
    }
  }

  // Notify the demuxer of an operation that started and finished immediately.
  template <typename Handler, typename Completion_Context>
  void operation_immediate(Handler handler, Completion_Context context,
      bool allow_nested_delivery)
  {
    operation_started();
    operation_completed(handler, context, allow_nested_delivery);
  }

private:
  // The demuxer that owns this service.
  demuxer_type& demuxer_;

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
