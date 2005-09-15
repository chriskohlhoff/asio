//
// win_iocp_demuxer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
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

// This service is only supported on Win32 (NT4 and later).
#if defined(_WIN32) && defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)

// Define this to indicate that IOCP is supported on the target platform.
#define ASIO_HAS_IOCP_DEMUXER 1

#include "asio/detail/push_options.hpp"
#include <memory>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/demuxer_run_call_stack.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_iocp_operation.hpp"

namespace asio {
namespace detail {

class win_iocp_demuxer_service
{
public:
  // Constructor.
  template <typename Demuxer>
  win_iocp_demuxer_service(Demuxer& demuxer)
    : iocp_(::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0, 0)),
      outstanding_work_(0),
      interrupted_(0)
  {
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

    demuxer_run_call_stack<win_iocp_demuxer_service>::context ctx(this);

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
        // Dispatch the operation.
        win_iocp_operation* op = static_cast<win_iocp_operation*>(overlapped);
        op->do_completion(last_error, bytes_transferred);
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
    : public win_iocp_operation
  {
    handler_operation(win_iocp_demuxer_service& demuxer_service,
        Handler handler)
      : win_iocp_operation(&handler_operation<Handler>::do_completion_impl),
        demuxer_service_(demuxer_service),
        handler_(handler)
    {
      demuxer_service_.work_started();
    }

    ~handler_operation()
    {
      demuxer_service_.work_finished();
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

  private:
    // Prevent copying and assignment.
    handler_operation(const handler_operation&);
    void operator=(const handler_operation&);
    
    static void do_completion_impl(win_iocp_operation* op, DWORD, size_t)
    {
      std::auto_ptr<handler_operation<Handler> > h(
          static_cast<handler_operation<Handler>*>(op));
      do_upcall(h->handler_);
    }

    win_iocp_demuxer_service& demuxer_service_;
    Handler handler_;
  };

  // Request the demuxer to invoke the given handler.
  template <typename Handler>
  void dispatch(Handler handler)
  {
    if (demuxer_run_call_stack<win_iocp_demuxer_service>::contains(this))
      handler_operation<Handler>::do_upcall(handler);
    else
      post(handler);
  }

  // Request the demuxer to invoke the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler)
  {
    win_iocp_operation* op = new handler_operation<Handler>(*this, handler);
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

  // The count of unfinished work.
  long outstanding_work_;

  // Flag to indicate whether the event loop has been interrupted.
  long interrupted_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32) && defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_DEMUXER_SERVICE_HPP
