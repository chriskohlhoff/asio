//
// win_iocp_demuxer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_DEMUXER_SERVICE_HPP
#define ASIO_DETAIL_WIN_IOCP_DEMUXER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

// This service is only supported on Win32 (NT4 and later).
#if defined(BOOST_WINDOWS)
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)

// Define this to indicate that IOCP is supported on the target platform.
#define ASIO_HAS_IOCP_DEMUXER 1

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include <memory>
#include "asio/detail/pop_options.hpp"

#include "asio/system_exception.hpp"
#include "asio/detail/demuxer_run_call_stack.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_iocp_operation.hpp"

namespace asio {
namespace detail {

template <typename Allocator>
class win_iocp_demuxer_service
{
public:
  // Base class for all operations.
  typedef win_iocp_operation<Allocator> operation;

  // Constructor.
  template <typename Demuxer>
  win_iocp_demuxer_service(Demuxer& demuxer)
    : iocp_(::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0, 0)),
      outstanding_work_(0),
      interrupted_(0),
      allocator_(demuxer.get_allocator())
  {
    if (!iocp_.handle)
    {
      DWORD last_error = ::GetLastError();
      system_exception e("iocp", last_error);
      boost::throw_exception(e);
    }
  }

  // Register a socket with the demuxer.
  void register_socket(socket_type sock)
  {
    HANDLE sock_as_handle = reinterpret_cast<HANDLE>(sock);
    ::CreateIoCompletionPort(sock_as_handle, iocp_.handle, 0, 0);
  }

  // Run the demuxer's event processing loop.
  void run()
  {
    if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
      return;

    typedef demuxer_run_call_stack<win_iocp_demuxer_service<Allocator> > drcs;
    typename drcs::context ctx(this);

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
        // Ensure that the demuxer does not exit due to running out of work
        // while we make the upcall.
        struct auto_work
        {
          auto_work(win_iocp_demuxer_service& demuxer_service)
            : demuxer_service_(demuxer_service)
          {
            demuxer_service_.work_started();
          }

          ~auto_work()
          {
            demuxer_service_.work_finished();
          }

        private:
          win_iocp_demuxer_service& demuxer_service_;
        } work(*this);

        // Dispatch the operation.
        operation* op = static_cast<operation*>(overlapped);
        op->do_completion(last_error, bytes_transferred, allocator_);
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

  // Interrupt the demuxer's event processing loop.
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

  // Reset the demuxer in preparation for a subsequent run invocation.
  void reset()
  {
    ::InterlockedExchange(&interrupted_, 0);
  }

  // Notify the demuxer that some work has started.
  void work_started()
  {
    ::InterlockedIncrement(&outstanding_work_);
  }

  // Notify the demuxer that some work has finished.
  void work_finished()
  {
    if (::InterlockedDecrement(&outstanding_work_) == 0)
      interrupt();
  }

  template <typename Handler>
  struct handler_operation
    : public operation
  {
    handler_operation(win_iocp_demuxer_service& demuxer_service,
        Handler handler)
      : operation(&handler_operation<Handler>::do_completion_impl),
        demuxer_service_(demuxer_service),
        handler_(handler)
    {
      demuxer_service_.work_started();
    }

    ~handler_operation()
    {
      demuxer_service_.work_finished();
    }

  private:
    // Prevent copying and assignment.
    handler_operation(const handler_operation&);
    void operator=(const handler_operation&);
    
    static void do_completion_impl(operation* op, DWORD, size_t,
        const Allocator& void_allocator)
    {
      // Take ownership of the operation object.
      typedef handler_operation<Handler> op_type;
      op_type* handler_op(static_cast<op_type*>(op));
      typedef handler_alloc_traits<Handler, op_type, Allocator> alloc_traits;
      handler_ptr<alloc_traits> ptr(handler_op->handler_,
          void_allocator, handler_op);

      // Make a copy of the handler so that the memory can be deallocated before
      // the upcall is made.
      Handler handler(handler_op->handler_);

      // Free the memory associated with the handler.
      ptr.reset();

      // Make the upcall.
      handler();
    }

    win_iocp_demuxer_service& demuxer_service_;
    Handler handler_;
  };

  // Request the demuxer to invoke the given handler.
  template <typename Handler>
  void dispatch(Handler handler)
  {
    if (demuxer_run_call_stack<win_iocp_demuxer_service>::contains(this))
      handler();
    else
      post(handler);
  }

  // Request the demuxer to invoke the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef handler_operation<Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type, Allocator> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler, allocator_);
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

  // The allocator to be used for allocating dynamic objects.
  Allocator allocator_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)
#endif // defined(BOOST_WINDOWS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_DEMUXER_SERVICE_HPP
