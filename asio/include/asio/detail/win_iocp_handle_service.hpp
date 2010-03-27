//
// win_iocp_handle_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2008 Rep Invariant Systems, Inc. (info@repinvariant.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP
#define ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/win_iocp_io_service_fwd.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/push_options.hpp"
#include <boost/cstdint.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/operation.hpp"
#include "asio/detail/win_iocp_io_service.hpp"

namespace asio {
namespace detail {

class win_iocp_handle_service
{
public:
  // The native type of a stream handle.
  typedef HANDLE native_type;

  // The implementation type of the stream handle.
  class implementation_type
  {
  public:
    // Default constructor.
    implementation_type()
      : handle_(INVALID_HANDLE_VALUE),
        safe_cancellation_thread_id_(0),
        next_(0),
        prev_(0)
    {
    }

  private:
    // Only this service will have access to the internal values.
    friend class win_iocp_handle_service;

    // The native stream handle representation.
    native_type handle_;

    // The ID of the thread from which it is safe to cancel asynchronous
    // operations. 0 means no asynchronous operations have been started yet.
    // ~0 means asynchronous operations have been started from more than one
    // thread, and cancellation is not supported for the handle.
    DWORD safe_cancellation_thread_id_;

    // Pointers to adjacent handle implementations in linked list.
    implementation_type* next_;
    implementation_type* prev_;
  };

  win_iocp_handle_service(asio::io_service& io_service)
    : iocp_service_(asio::use_service<win_iocp_io_service>(io_service)),
      mutex_(),
      impl_list_(0)
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    // Close all implementations, causing all operations to complete.
    asio::detail::mutex::scoped_lock lock(mutex_);
    implementation_type* impl = impl_list_;
    while (impl)
    {
      close_for_destruction(*impl);
      impl = impl->next_;
    }
  }

  // Construct a new handle implementation.
  void construct(implementation_type& impl)
  {
    impl.handle_ = INVALID_HANDLE_VALUE;
    impl.safe_cancellation_thread_id_ = 0;

    // Insert implementation into linked list of all implementations.
    asio::detail::mutex::scoped_lock lock(mutex_);
    impl.next_ = impl_list_;
    impl.prev_ = 0;
    if (impl_list_)
      impl_list_->prev_ = &impl;
    impl_list_ = &impl;
  }

  // Destroy a handle implementation.
  void destroy(implementation_type& impl)
  {
    close_for_destruction(impl);
    
    // Remove implementation from linked list of all implementations.
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (impl_list_ == &impl)
      impl_list_ = impl.next_;
    if (impl.prev_)
      impl.prev_->next_ = impl.next_;
    if (impl.next_)
      impl.next_->prev_= impl.prev_;
    impl.next_ = 0;
    impl.prev_ = 0;
  }

  // Assign a native handle to a handle implementation.
  asio::error_code assign(implementation_type& impl,
      const native_type& native_handle, asio::error_code& ec)
  {
    if (is_open(impl))
    {
      ec = asio::error::already_open;
      return ec;
    }

    if (iocp_service_.register_handle(native_handle, ec))
      return ec;

    impl.handle_ = native_handle;
    ec = asio::error_code();
    return ec;
  }

  // Determine whether the handle is open.
  bool is_open(const implementation_type& impl) const
  {
    return impl.handle_ != INVALID_HANDLE_VALUE;
  }

  // Destroy a handle implementation.
  asio::error_code close(implementation_type& impl,
      asio::error_code& ec)
  {
    if (is_open(impl))
    {
      if (!::CloseHandle(impl.handle_))
      {
        DWORD last_error = ::GetLastError();
        ec = asio::error_code(last_error,
            asio::error::get_system_category());
        return ec;
      }

      impl.handle_ = INVALID_HANDLE_VALUE;
      impl.safe_cancellation_thread_id_ = 0;
    }

    ec = asio::error_code();
    return ec;
  }

  // Get the native handle representation.
  native_type native(const implementation_type& impl) const
  {
    return impl.handle_;
  }

  // Cancel all operations associated with the handle.
  asio::error_code cancel(implementation_type& impl,
      asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
    }
    else if (FARPROC cancel_io_ex_ptr = ::GetProcAddress(
          ::GetModuleHandleA("KERNEL32"), "CancelIoEx"))
    {
      // The version of Windows supports cancellation from any thread.
      typedef BOOL (WINAPI* cancel_io_ex_t)(HANDLE, LPOVERLAPPED);
      cancel_io_ex_t cancel_io_ex = (cancel_io_ex_t)cancel_io_ex_ptr;
      if (!cancel_io_ex(impl.handle_, 0))
      {
        DWORD last_error = ::GetLastError();
        if (last_error == ERROR_NOT_FOUND)
        {
          // ERROR_NOT_FOUND means that there were no operations to be
          // cancelled. We swallow this error to match the behaviour on other
          // platforms.
          ec = asio::error_code();
        }
        else
        {
          ec = asio::error_code(last_error,
              asio::error::get_system_category());
        }
      }
      else
      {
        ec = asio::error_code();
      }
    }
    else if (impl.safe_cancellation_thread_id_ == 0)
    {
      // No operations have been started, so there's nothing to cancel.
      ec = asio::error_code();
    }
    else if (impl.safe_cancellation_thread_id_ == ::GetCurrentThreadId())
    {
      // Asynchronous operations have been started from the current thread only,
      // so it is safe to try to cancel them using CancelIo.
      if (!::CancelIo(impl.handle_))
      {
        DWORD last_error = ::GetLastError();
        ec = asio::error_code(last_error,
            asio::error::get_system_category());
      }
      else
      {
        ec = asio::error_code();
      }
    }
    else
    {
      // Asynchronous operations have been started from more than one thread,
      // so cancellation is not safe.
      ec = asio::error::operation_not_supported;
    }

    return ec;
  }

  class overlapped_wrapper
    : public OVERLAPPED
  {
  public:
    explicit overlapped_wrapper(asio::error_code& ec)
    {
      Internal = 0;
      InternalHigh = 0;
      Offset = 0;
      OffsetHigh = 0;

      // Create a non-signalled manual-reset event, for GetOverlappedResult.
      hEvent = ::CreateEvent(0, TRUE, FALSE, 0);
      if (hEvent)
      {
        // As documented in GetQueuedCompletionStatus, setting the low order
        // bit of this event prevents our synchronous writes from being treated
        // as completion port events.
        *reinterpret_cast<DWORD_PTR*>(&hEvent) |= 1;
      }
      else
      {
        DWORD last_error = ::GetLastError();
        ec = asio::error_code(last_error,
            asio::error::get_system_category());
      }
    }

    ~overlapped_wrapper()
    {
      if (hEvent)
      {
        ::CloseHandle(hEvent);
      }
    }
  };

  // Write the given data. Returns the number of bytes written.
  template <typename ConstBufferSequence>
  size_t write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    return write_some_at(impl, 0, buffers, ec);
  }

  // Write the given data at the specified offset. Returns the number of bytes
  // written.
  template <typename ConstBufferSequence>
  size_t write_some_at(implementation_type& impl, boost::uint64_t offset,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return 0;
    }

    asio::const_buffer buffer =
      buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence>::first(buffers);

    // A request to write 0 bytes on a handle is a no-op.
    if (asio::buffer_size(buffer) == 0)
    {
      ec = asio::error_code();
      return 0;
    }

    overlapped_wrapper overlapped(ec);
    if (ec)
    {
      return 0;
    }

    // Write the data. 
    overlapped.Offset = offset & 0xFFFFFFFF;
    overlapped.OffsetHigh = (offset >> 32) & 0xFFFFFFFF;
    BOOL ok = ::WriteFile(impl.handle_,
        asio::buffer_cast<LPCVOID>(buffer),
        static_cast<DWORD>(asio::buffer_size(buffer)), 0, &overlapped);
    if (!ok) 
    {
      DWORD last_error = ::GetLastError();
      if (last_error != ERROR_IO_PENDING)
      {
        ec = asio::error_code(last_error,
            asio::error::get_system_category());
        return 0;
      }
    }

    // Wait for the operation to complete.
    DWORD bytes_transferred = 0;
    ok = ::GetOverlappedResult(impl.handle_,
        &overlapped, &bytes_transferred, TRUE);
    if (!ok)
    {
      DWORD last_error = ::GetLastError();
      ec = asio::error_code(last_error,
          asio::error::get_system_category());
      return 0;
    }

    ec = asio::error_code();
    return bytes_transferred;
  }

  template <typename ConstBufferSequence, typename Handler>
  class write_op : public operation
  {
  public:
    write_op(const ConstBufferSequence& buffers, Handler handler)
      : operation(&write_op::do_complete),
        buffers_(buffers),
        handler_(handler)
    {
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code ec, std::size_t bytes_transferred)
    {
      // Take ownership of the operation object.
      write_op* o(static_cast<write_op*>(base));
      typedef handler_alloc_traits<Handler, write_op> alloc_traits;
      handler_ptr<alloc_traits> ptr(o->handler_, o);

      // Make the upcall if required.
      if (owner)
      {
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        // Check whether buffers are still valid.
        buffer_sequence_adapter<asio::const_buffer,
            ConstBufferSequence>::validate(o->buffers_);
#endif // defined(ASIO_ENABLE_BUFFER_DEBUGGING)

        // Make a copy of the handler so that the memory can be deallocated
        // before the upcall is made. Even if we're not about to make an
        // upcall, a sub-object of the handler may be the true owner of the
        // memory associated with the handler. Consequently, a local copy of
        // the handler is required to ensure that any owning sub-object remains
        // valid until after we have deallocated the memory here.
        detail::binder2<Handler, asio::error_code, std::size_t>
          handler(o->handler_, ec, bytes_transferred);
        ptr.reset();
        asio::detail::fenced_block b;
        asio_handler_invoke_helpers::invoke(handler, handler);
      }
    }

  private:
    ConstBufferSequence buffers_;
    Handler handler_;
  };

  // Start an asynchronous write. The data being written must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, Handler handler)
  {
    async_write_some_at(impl, 0, buffers, handler);
  }

  // Start an asynchronous write at a specified offset. The data being written
  // must be valid for the lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_write_some_at(implementation_type& impl, boost::uint64_t offset,
      const ConstBufferSequence& buffers, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef write_op<ConstBufferSequence, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, buffers, handler);

    start_write_op(impl, offset,
        buffer_sequence_adapter<asio::const_buffer,
          ConstBufferSequence>::first(buffers), ptr.get());
    ptr.release();
  }

  // Read some data. Returns the number of bytes received.
  template <typename MutableBufferSequence>
  size_t read_some(implementation_type& impl,
      const MutableBufferSequence& buffers, asio::error_code& ec)
  {
    return read_some_at(impl, 0, buffers, ec);
  }

  // Read some data at a specified offset. Returns the number of bytes received.
  template <typename MutableBufferSequence>
  size_t read_some_at(implementation_type& impl, boost::uint64_t offset,
      const MutableBufferSequence& buffers, asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return 0;
    }
    
    asio::mutable_buffer buffer =
      buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence>::first(buffers);

    // A request to read 0 bytes on a stream handle is a no-op.
    if (asio::buffer_size(buffer) == 0)
    {
      ec = asio::error_code();
      return 0;
    }

    overlapped_wrapper overlapped(ec);
    if (ec)
    {
      return 0;
    }

    // Read some data.
    overlapped.Offset = offset & 0xFFFFFFFF;
    overlapped.OffsetHigh = (offset >> 32) & 0xFFFFFFFF;
    BOOL ok = ::ReadFile(impl.handle_,
        asio::buffer_cast<LPVOID>(buffer),
        static_cast<DWORD>(asio::buffer_size(buffer)), 0, &overlapped);
    if (!ok) 
    {
      DWORD last_error = ::GetLastError();
      if (last_error != ERROR_IO_PENDING && last_error != ERROR_MORE_DATA)
      {
        if (last_error == ERROR_HANDLE_EOF)
        {
          ec = asio::error::eof;
        }
        else
        {
          ec = asio::error_code(last_error,
              asio::error::get_system_category());
        }
        return 0;
      }
    }

    // Wait for the operation to complete.
    DWORD bytes_transferred = 0;
    ok = ::GetOverlappedResult(impl.handle_,
        &overlapped, &bytes_transferred, TRUE);
    if (!ok)
    {
      DWORD last_error = ::GetLastError();
      if (last_error == ERROR_HANDLE_EOF)
      {
        ec = asio::error::eof;
      }
      else
      {
        ec = asio::error_code(last_error,
            asio::error::get_system_category());
      }
      return 0;
    }

    ec = asio::error_code();
    return bytes_transferred;
  }

  template <typename MutableBufferSequence, typename Handler>
  class read_op : public operation
  {
  public:
    read_op(const MutableBufferSequence& buffers, Handler handler)
      : operation(&read_op::do_complete),
        buffers_(buffers),
        handler_(handler)
    {
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code ec, std::size_t bytes_transferred)
    {
      // Take ownership of the operation object.
      read_op* o(static_cast<read_op*>(base));
      typedef handler_alloc_traits<Handler, read_op> alloc_traits;
      handler_ptr<alloc_traits> ptr(o->handler_, o);

      // Make the upcall if required.
      if (owner)
      {
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        // Check whether buffers are still valid.
        buffer_sequence_adapter<asio::mutable_buffer,
            MutableBufferSequence>::validate(o->buffers_);
#endif // defined(ASIO_ENABLE_BUFFER_DEBUGGING)

        // Map non-portable errors to their portable counterparts.
        if (ec.value() == ERROR_HANDLE_EOF)
        {
          ec = asio::error::eof;
        }

        // Make a copy of the handler so that the memory can be deallocated
        // before the upcall is made. Even if we're not about to make an
        // upcall, a sub-object of the handler may be the true owner of the
        // memory associated with the handler. Consequently, a local copy of
        // the handler is required to ensure that any owning sub-object remains
        // valid until after we have deallocated the memory here.
        detail::binder2<Handler, asio::error_code, std::size_t>
          handler(o->handler_, ec, bytes_transferred);
        ptr.reset();
        asio::detail::fenced_block b;
        asio_handler_invoke_helpers::invoke(handler, handler);
      }
    }

  private:
    MutableBufferSequence buffers_;
    Handler handler_;
  };

  // Start an asynchronous read. The buffer for the data being received must be
  // valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_read_some(implementation_type& impl,
      const MutableBufferSequence& buffers, Handler handler)
  {
    async_read_some_at(impl, 0, buffers, handler);
  }

  // Start an asynchronous read at a specified offset. The buffer for the data
  // being received must be valid for the lifetime of the asynchronous
  // operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_read_some_at(implementation_type& impl, boost::uint64_t offset,
      const MutableBufferSequence& buffers, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef read_op<MutableBufferSequence, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, buffers, handler);

    start_read_op(impl, offset,
        buffer_sequence_adapter<asio::mutable_buffer,
          MutableBufferSequence>::first(buffers), ptr.get());
    ptr.release();
  }

private:
  // Prevent the use of the null_buffers type with this service.
  size_t write_some(implementation_type& impl,
      const null_buffers& buffers, asio::error_code& ec);
  size_t write_some_at(implementation_type& impl, boost::uint64_t offset,
      const null_buffers& buffers, asio::error_code& ec);
  template <typename Handler>
  void async_write_some(implementation_type& impl,
      const null_buffers& buffers, Handler handler);
  template <typename Handler>
  void async_write_some_at(implementation_type& impl, boost::uint64_t offset,
      const null_buffers& buffers, Handler handler);
  size_t read_some(implementation_type& impl,
      const null_buffers& buffers, asio::error_code& ec);
  size_t read_some_at(implementation_type& impl, boost::uint64_t offset,
      const null_buffers& buffers, asio::error_code& ec);
  template <typename Handler>
  void async_read_some(implementation_type& impl,
      const null_buffers& buffers, Handler handler);
  template <typename Handler>
  void async_read_some_at(implementation_type& impl, boost::uint64_t offset,
      const null_buffers& buffers, Handler handler);

  // Helper function to start a write operation.
  void start_write_op(implementation_type& impl, boost::uint64_t offset,
      const asio::const_buffer& buffer, operation* op)
  {
    update_cancellation_thread_id(impl);
    iocp_service_.work_started();

    if (!is_open(impl))
    {
      iocp_service_.on_completion(op, asio::error::bad_descriptor);
    }
    else if (asio::buffer_size(buffer) == 0)
    {
      // A request to write 0 bytes on a handle is a no-op.
      iocp_service_.on_completion(op);
    }
    else
    {
      DWORD bytes_transferred = 0;
      op->Offset = offset & 0xFFFFFFFF;
      op->OffsetHigh = (offset >> 32) & 0xFFFFFFFF;
      BOOL ok = ::WriteFile(impl.handle_,
          asio::buffer_cast<LPCVOID>(buffer),
          static_cast<DWORD>(asio::buffer_size(buffer)),
          &bytes_transferred, op);
      DWORD last_error = ::GetLastError();
      if (!ok && last_error != ERROR_IO_PENDING
          && last_error != ERROR_MORE_DATA)
      {
        iocp_service_.on_completion(op, last_error, bytes_transferred);
      }
      else
      {
        iocp_service_.on_pending(op);
      }
    }
  }

  // Helper function to start a read operation.
  void start_read_op(implementation_type& impl, boost::uint64_t offset,
      const asio::mutable_buffer& buffer, operation* op)
  {
    update_cancellation_thread_id(impl);
    iocp_service_.work_started();

    if (!is_open(impl))
    {
      iocp_service_.on_completion(op, asio::error::bad_descriptor);
    }
    else if (asio::buffer_size(buffer) == 0)
    {
      // A request to read 0 bytes on a handle is a no-op.
      iocp_service_.on_completion(op);
    }
    else
    {
      DWORD bytes_transferred = 0;
      op->Offset = offset & 0xFFFFFFFF;
      op->OffsetHigh = (offset >> 32) & 0xFFFFFFFF;
      BOOL ok = ::ReadFile(impl.handle_,
          asio::buffer_cast<LPVOID>(buffer),
          static_cast<DWORD>(asio::buffer_size(buffer)),
          &bytes_transferred, op);
      DWORD last_error = ::GetLastError();
      if (!ok && last_error != ERROR_IO_PENDING
          && last_error != ERROR_MORE_DATA)
      {
        iocp_service_.on_completion(op, last_error, bytes_transferred);
      }
      else
      {
        iocp_service_.on_pending(op);
      }
    }
  }

  // Update the ID of the thread from which cancellation is safe.
  void update_cancellation_thread_id(implementation_type& impl)
  {
#if defined(ASIO_ENABLE_CANCELIO)
    if (impl.safe_cancellation_thread_id_ == 0)
      impl.safe_cancellation_thread_id_ = ::GetCurrentThreadId();
    else if (impl.safe_cancellation_thread_id_ != ::GetCurrentThreadId())
      impl.safe_cancellation_thread_id_ = ~DWORD(0);
#else // defined(ASIO_ENABLE_CANCELIO)
    (void)impl;
#endif // defined(ASIO_ENABLE_CANCELIO)
  }

  // Helper function to close a handle when the associated object is being
  // destroyed.
  void close_for_destruction(implementation_type& impl)
  {
    if (is_open(impl))
    {
      ::CloseHandle(impl.handle_);
      impl.handle_ = INVALID_HANDLE_VALUE;
      impl.safe_cancellation_thread_id_ = 0;
    }
  }

  // The IOCP service used for running asynchronous operations and dispatching
  // handlers.
  win_iocp_io_service& iocp_service_;

  // Mutex to protect access to the linked list of implementations.
  asio::detail::mutex mutex_;

  // The head of a linked list of all implementations.
  implementation_type* impl_list_;
};

} // namespace detail
} // namespace asio

#endif // defined(ASIO_HAS_IOCP)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP
