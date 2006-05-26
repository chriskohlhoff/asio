//
// win_iocp_io_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/io_service.hpp"
#include "asio/system_exception.hpp"
#include "asio/detail/call_stack.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_iocp_operation.hpp"

namespace asio {
namespace detail {

class win_iocp_io_service
  : public asio::io_service::service
{
public:
  // Base class for all operations.
  typedef win_iocp_operation operation;

  // Constructor.
  win_iocp_io_service(asio::io_service& io_service)
    : asio::io_service::service(io_service),
      iocp_(::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0, 0)),
      outstanding_work_(0),
      interrupted_(0),
      shutdown_(0)
  {
    if (!iocp_.handle)
    {
      DWORD last_error = ::GetLastError();
      system_exception e("iocp", last_error);
      boost::throw_exception(e);
    }
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    ::InterlockedExchange(&shutdown_, 1);

    for (;;)
    {
      DWORD bytes_transferred = 0;
#if (WINVER < 0x0500)
      DWORD completion_key = 0;
#else
      DWORD_PTR completion_key = 0;
#endif
      LPOVERLAPPED overlapped = 0;
      ::GetQueuedCompletionStatus(iocp_.handle,
          &bytes_transferred, &completion_key, &overlapped, 0);
      DWORD last_error = ::GetLastError();
      if (last_error == WAIT_TIMEOUT)
        break;
      if (overlapped)
        static_cast<operation*>(overlapped)->destroy();
    }
  }

  // Register a socket with the IO completion port.
  void register_socket(socket_type sock)
  {
    HANDLE sock_as_handle = reinterpret_cast<HANDLE>(sock);
    ::CreateIoCompletionPort(sock_as_handle, iocp_.handle, 0, 0);
  }

  struct auto_work
  {
    auto_work(win_iocp_io_service& io_service)
      : io_service_(io_service)
    {
      io_service_.work_started();
    }

    ~auto_work()
    {
      io_service_.work_finished();
    }

  private:
    win_iocp_io_service& io_service_;
  };

  // Run the event processing loop.
  void run()
  {
    if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
      return;

    call_stack<win_iocp_io_service>::context ctx(this);

    for (;;)
    {
      // Get the next operation from the queue.
      DWORD bytes_transferred = 0;
#if (WINVER < 0x0500)
      DWORD completion_key = 0;
#else
      DWORD_PTR completion_key = 0;
#endif
      LPOVERLAPPED overlapped = 0;
      ::SetLastError(0);
      ::GetQueuedCompletionStatus(iocp_.handle, &bytes_transferred,
          &completion_key, &overlapped, INFINITE);
      DWORD last_error = ::GetLastError();

      if (overlapped)
      {
        // We may have been passed a last_error value in the completion_key.
        if (last_error == 0)
        {
          last_error = completion_key;
        }

        // Ensure that the io_service does not exit due to running out of work
        // while we make the upcall.
        auto_work work(*this);

        // Dispatch the operation.
        operation* op = static_cast<operation*>(overlapped);
        op->do_completion(last_error, bytes_transferred);
      }
      else
      {
        // The interrupted_ flag is always checked to ensure that any leftover
        // interrupts from a previous run invocation are ignored.
        if (::InterlockedExchangeAdd(&interrupted_, 0) != 0)
        {
          // Wake up next thread that is blocked on GetQueuedCompletionStatus.
          if (!::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0))
          {
            DWORD last_error = ::GetLastError();
            system_exception e("pqcs", last_error);
            boost::throw_exception(e);
          }
          break;
        }
      }
    }
  }

  // Interrupt the event processing loop.
  void interrupt()
  {
    if (::InterlockedExchange(&interrupted_, 1) == 0)
    {
      if (!::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0))
      {
        DWORD last_error = ::GetLastError();
        system_exception e("pqcs", last_error);
        boost::throw_exception(e);
      }
    }
  }

  // Reset in preparation for a subsequent run invocation.
  void reset()
  {
    ::InterlockedExchange(&interrupted_, 0);
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
      interrupt();
  }

  template <typename Handler>
  struct handler_operation
    : public operation
  {
    handler_operation(win_iocp_io_service& io_service,
        Handler handler)
      : operation(&handler_operation<Handler>::do_completion_impl,
          &handler_operation<Handler>::destroy_impl),
        io_service_(io_service),
        handler_(handler)
    {
      io_service_.work_started();
    }

    ~handler_operation()
    {
      io_service_.work_finished();
    }

  private:
    // Prevent copying and assignment.
    handler_operation(const handler_operation&);
    void operator=(const handler_operation&);
    
    static void do_completion_impl(operation* op, DWORD, size_t)
    {
      // Take ownership of the operation object.
      typedef handler_operation<Handler> op_type;
      op_type* handler_op(static_cast<op_type*>(op));
      typedef handler_alloc_traits<Handler, op_type> alloc_traits;
      handler_ptr<alloc_traits> ptr(handler_op->handler_, handler_op);

      // Make a copy of the handler so that the memory can be deallocated before
      // the upcall is made.
      Handler handler(handler_op->handler_);

      // Free the memory associated with the handler.
      ptr.reset();

      // Make the upcall.
      handler();
    }

    static void destroy_impl(operation* op)
    {
      // Take ownership of the operation object.
      typedef handler_operation<Handler> op_type;
      op_type* handler_op(static_cast<op_type*>(op));
      typedef handler_alloc_traits<Handler, op_type> alloc_traits;
      handler_ptr<alloc_traits> ptr(handler_op->handler_, handler_op);
    }

    win_iocp_io_service& io_service_;
    Handler handler_;
  };

  // Request invocation of the given handler.
  template <typename Handler>
  void dispatch(Handler handler)
  {
    if (call_stack<win_iocp_io_service>::contains(this))
      handler();
    else
      post(handler);
  }

  // Request invocation of the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler)
  {
    // If the service has been shut down we silently discard the handler.
    if (::InterlockedExchangeAdd(&shutdown_, 0) != 0)
      return;

    // Allocate and construct an operation to wrap the handler.
    typedef handler_operation<Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, *this, handler);

    // Enqueue the operation on the I/O completion port.
    if (!::PostQueuedCompletionStatus(iocp_.handle, 0, 0, ptr.get()))
    {
      DWORD last_error = ::GetLastError();
      system_exception e("pqcs", last_error);
      boost::throw_exception(e);
    }

    // Operation has been successfully posted.
    ptr.release();
  }

  // Request invocation of the given OVERLAPPED-derived operation.
  void post_completion(win_iocp_operation* op, DWORD op_last_error,
      DWORD bytes_transferred)
  {
    // Enqueue the operation on the I/O completion port.
    if (!::PostQueuedCompletionStatus(iocp_.handle,
          bytes_transferred, op_last_error, op))
    {
      DWORD last_error = ::GetLastError();
      system_exception e("pqcs", last_error);
      boost::throw_exception(e);
    }
  }

private:
  // The IO completion port used for queueing operations.
  struct iocp_holder
  {
    HANDLE handle;
    iocp_holder(HANDLE h) : handle(h) {}
    ~iocp_holder() { ::CloseHandle(handle); }
  } iocp_;

  // The count of unfinished work.
  long outstanding_work_;

  // Flag to indicate whether the event loop has been interrupted.
  long interrupted_;

  // Flag to indicate whether the service has been shut down.
  long shutdown_;
};

} // namespace detail
} // namespace asio

#endif // defined(ASIO_HAS_IOCP)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_IO_SERVICE_HPP
