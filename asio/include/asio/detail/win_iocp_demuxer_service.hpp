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

#if defined(_WIN32) // This service is only supported on Win32

#include "asio/basic_demuxer.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/tss_bool.hpp"
#include "asio/detail/win_iocp_operation.hpp"

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
      outstanding_work_(0),
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

  // Run the demuxer's event processing loop.
  void run()
  {
    if (::InterlockedExchangeAdd(&outstanding_work_, 0) == 0)
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
        win_iocp_operation* op = static_cast<win_iocp_operation*>(overlapped);
        op->do_completion(*this, iocp_.handle, last_error, bytes_transferred);
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
    handler_operation(Handler handler)
      : win_iocp_operation(&handler_operation<Handler>::do_completion_impl),
        handler_(handler)
    {
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
    static void do_completion_impl(win_iocp_operation* op,
        win_iocp_demuxer_service& demuxer_service, HANDLE iocp, DWORD, size_t)
    {
      handler_operation<Handler>* h =
        static_cast<handler_operation<Handler>*>(op);
      do_upcall(h->handler_);
      demuxer_service.work_finished();
      delete h;
    }

    Handler handler_;
  };

  // Request the demuxer to invoke the given handler.
  template <typename Handler>
  void dispatch(Handler handler)
  {
    if (current_thread_in_pool_)
      handler_operation<Handler>::do_upcall(handler);
    else
      post(handler);
  }

  // Request the demuxer to invoke the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler)
  {
    win_iocp_operation* op = new handler_operation<Handler>(handler);
    work_started();
    ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, op);
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

  // The count of unfinished work.
  long outstanding_work_;

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
