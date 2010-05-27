//
// detail/win_iocp_socket_service_base.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_BASE_HPP
#define ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include <cstring>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/addressof.hpp>
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/null_buffers_op.hpp"
#include "asio/detail/operation.hpp"
#include "asio/detail/reactor.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/shared_ptr.hpp"
#include "asio/detail/socket_connect_op.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/weak_ptr.hpp"
#include "asio/detail/win_iocp_io_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class win_iocp_socket_service_base
{
public:
  struct noop_deleter { void operator()(void*) {} };
  typedef shared_ptr<void> shared_cancel_token_type;
  typedef weak_ptr<void> weak_cancel_token_type;

  // The implementation type of the socket.
  struct base_implementation_type
  {
    // The native socket representation.
    socket_type socket_;

    // The current state of the socket.
    socket_ops::state_type state_;

    // We use a shared pointer as a cancellation token here to work around the
    // broken Windows support for cancellation. MSDN says that when you call
    // closesocket any outstanding WSARecv or WSASend operations will complete
    // with the error ERROR_OPERATION_ABORTED. In practice they complete with
    // ERROR_NETNAME_DELETED, which means you can't tell the difference between
    // a local cancellation and the socket being hard-closed by the peer.
    shared_cancel_token_type cancel_token_;

    // Per-descriptor data used by the reactor.
    reactor::per_descriptor_data reactor_data_;

#if defined(ASIO_ENABLE_CANCELIO)
    // The ID of the thread from which it is safe to cancel asynchronous
    // operations. 0 means no asynchronous operations have been started yet.
    // ~0 means asynchronous operations have been started from more than one
    // thread, and cancellation is not supported for the socket.
    DWORD safe_cancellation_thread_id_;
#endif // defined(ASIO_ENABLE_CANCELIO)

    // Pointers to adjacent socket implementations in linked list.
    base_implementation_type* next_;
    base_implementation_type* prev_;
  };

  // Constructor.
  win_iocp_socket_service_base(asio::io_service& io_service)
    : io_service_(io_service),
      iocp_service_(use_service<win_iocp_io_service>(io_service)),
      reactor_(0),
      mutex_(),
      impl_list_(0)
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    // Close all implementations, causing all operations to complete.
    asio::detail::mutex::scoped_lock lock(mutex_);
    base_implementation_type* impl = impl_list_;
    while (impl)
    {
      asio::error_code ignored_ec;
      close_for_destruction(*impl);
      impl = impl->next_;
    }
  }

  // Construct a new socket implementation.
  void construct(base_implementation_type& impl)
  {
    impl.socket_ = invalid_socket;
    impl.state_ = 0;
    impl.cancel_token_.reset();
#if defined(ASIO_ENABLE_CANCELIO)
    impl.safe_cancellation_thread_id_ = 0;
#endif // defined(ASIO_ENABLE_CANCELIO)

    // Insert implementation into linked list of all implementations.
    asio::detail::mutex::scoped_lock lock(mutex_);
    impl.next_ = impl_list_;
    impl.prev_ = 0;
    if (impl_list_)
      impl_list_->prev_ = &impl;
    impl_list_ = &impl;
  }

  // Destroy a socket implementation.
  void destroy(base_implementation_type& impl)
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

  // Determine whether the socket is open.
  bool is_open(const base_implementation_type& impl) const
  {
    return impl.socket_ != invalid_socket;
  }

  // Destroy a socket implementation.
  asio::error_code close(base_implementation_type& impl,
      asio::error_code& ec)
  {
    if (is_open(impl))
    {
      // Check if the reactor was created, in which case we need to close the
      // socket on the reactor as well to cancel any operations that might be
      // running there.
      reactor* r = static_cast<reactor*>(
            interlocked_compare_exchange_pointer(
              reinterpret_cast<void**>(&reactor_), 0, 0));
      if (r)
        r->close_descriptor(impl.socket_, impl.reactor_data_);
    }

    if (socket_ops::close(impl.socket_, impl.state_, false, ec) == 0)
    {
      impl.socket_ = invalid_socket;
      impl.state_ = 0;
      impl.cancel_token_.reset();
#if defined(ASIO_ENABLE_CANCELIO)
      impl.safe_cancellation_thread_id_ = 0;
#endif // defined(ASIO_ENABLE_CANCELIO)
    }

    return ec;
  }

  // Cancel all operations associated with the socket.
  asio::error_code cancel(base_implementation_type& impl,
      asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return ec;
    }
    else if (FARPROC cancel_io_ex_ptr = ::GetProcAddress(
          ::GetModuleHandleA("KERNEL32"), "CancelIoEx"))
    {
      // The version of Windows supports cancellation from any thread.
      typedef BOOL (WINAPI* cancel_io_ex_t)(HANDLE, LPOVERLAPPED);
      cancel_io_ex_t cancel_io_ex = (cancel_io_ex_t)cancel_io_ex_ptr;
      socket_type sock = impl.socket_;
      HANDLE sock_as_handle = reinterpret_cast<HANDLE>(sock);
      if (!cancel_io_ex(sock_as_handle, 0))
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
#if defined(ASIO_ENABLE_CANCELIO)
    else if (impl.safe_cancellation_thread_id_ == 0)
    {
      // No operations have been started, so there's nothing to cancel.
      ec = asio::error_code();
    }
    else if (impl.safe_cancellation_thread_id_ == ::GetCurrentThreadId())
    {
      // Asynchronous operations have been started from the current thread only,
      // so it is safe to try to cancel them using CancelIo.
      socket_type sock = impl.socket_;
      HANDLE sock_as_handle = reinterpret_cast<HANDLE>(sock);
      if (!::CancelIo(sock_as_handle))
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
#else // defined(ASIO_ENABLE_CANCELIO)
    else
    {
      // Cancellation is not supported as CancelIo may not be used.
      ec = asio::error::operation_not_supported;
    }
#endif // defined(ASIO_ENABLE_CANCELIO)

    return ec;
  }

  // Determine whether the socket is at the out-of-band data mark.
  bool at_mark(const base_implementation_type& impl,
      asio::error_code& ec) const
  {
    return socket_ops::sockatmark(impl.socket_, ec);
  }

  // Determine the number of bytes available for reading.
  std::size_t available(const base_implementation_type& impl,
      asio::error_code& ec) const
  {
    return socket_ops::available(impl.socket_, ec);
  }

  // Place the socket into the state where it will listen for new connections.
  asio::error_code listen(base_implementation_type& impl, int backlog,
      asio::error_code& ec)
  {
    socket_ops::listen(impl.socket_, backlog, ec);
    return ec;
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command>
  asio::error_code io_control(base_implementation_type& impl,
      IO_Control_Command& command, asio::error_code& ec)
  {
    socket_ops::ioctl(impl.socket_, impl.state_, command.name(),
        static_cast<ioctl_arg_type*>(command.data()), ec);
    return ec;
  }

  /// Disable sends or receives on the socket.
  asio::error_code shutdown(base_implementation_type& impl,
      socket_base::shutdown_type what, asio::error_code& ec)
  {
    socket_ops::shutdown(impl.socket_, what, ec);
    return ec;
  }

  // Send the given data to the peer. Returns the number of bytes sent.
  template <typename ConstBufferSequence>
  size_t send(base_implementation_type& impl,
      const ConstBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence> bufs(buffers);

    return socket_ops::sync_send(impl.socket_, impl.state_,
        bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
  }

  // Wait until data can be sent without blocking.
  size_t send(base_implementation_type& impl, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    // Wait for socket to become ready.
    socket_ops::poll_write(impl.socket_, ec);

    return 0;
  }

  template <typename ConstBufferSequence, typename Handler>
  class send_op : public operation
  {
  public:
    send_op(weak_cancel_token_type cancel_token,
        const ConstBufferSequence& buffers, Handler handler)
      : operation(&send_op::do_complete),
        cancel_token_(cancel_token),
        buffers_(buffers),
        handler_(handler)
    {
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code ec, std::size_t bytes_transferred)
    {
      // Take ownership of the operation object.
      send_op* o(static_cast<send_op*>(base));
      typedef handler_alloc_traits<Handler, send_op> alloc_traits;
      handler_ptr<alloc_traits> ptr(o->handler_, o);

      // Make the upcall if required.
      if (owner)
      {
#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
        // Check whether buffers are still valid.
        buffer_sequence_adapter<asio::const_buffer,
            ConstBufferSequence>::validate(o->buffers_);
#endif // defined(ASIO_ENABLE_BUFFER_DEBUGGING)

        // Map non-portable errors to their portable counterparts.
        if (ec.value() == ERROR_NETNAME_DELETED)
        {
          if (o->cancel_token_.expired())
            ec = asio::error::operation_aborted;
          else
            ec = asio::error::connection_reset;
        }
        else if (ec.value() == ERROR_PORT_UNREACHABLE)
        {
          ec = asio::error::connection_refused;
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
    weak_cancel_token_type cancel_token_;
    ConstBufferSequence buffers_;
    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_send(base_implementation_type& impl,
      const ConstBufferSequence& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef send_op<ConstBufferSequence, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr,
        impl.cancel_token_, buffers, handler);

    buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence> bufs(buffers);

    start_send_op(impl, bufs.buffers(), bufs.count(), flags,
        (impl.state_ & socket_ops::stream_oriented) != 0 && bufs.all_empty(),
        ptr.get());
    ptr.release();
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler>
  void async_send(base_implementation_type& impl, const null_buffers&,
      socket_base::message_flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef null_buffers_op<Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(handler);

    start_reactor_op(impl, reactor::write_op, p.p);
    p.v = p.p = 0;
  }

  // Receive some data from the peer. Returns the number of bytes received.
  template <typename MutableBufferSequence>
  size_t receive(base_implementation_type& impl,
      const MutableBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(buffers);

    return socket_ops::sync_recv(impl.socket_, impl.state_,
        bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
  }

  // Wait until data can be received without blocking.
  size_t receive(base_implementation_type& impl, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    // Wait for socket to become ready.
    socket_ops::poll_read(impl.socket_, ec);

    return 0;
  }

  template <typename MutableBufferSequence, typename Handler>
  class receive_op : public operation
  {
  public:
    receive_op(socket_ops::state_type state,
        weak_cancel_token_type cancel_token,
        const MutableBufferSequence& buffers, Handler handler)
      : operation(&receive_op::do_complete),
        state_(state),
        cancel_token_(cancel_token),
        buffers_(buffers),
        handler_(handler)
    {
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code ec, std::size_t bytes_transferred)
    {
      // Take ownership of the operation object.
      receive_op* o(static_cast<receive_op*>(base));
      typedef handler_alloc_traits<Handler, receive_op> alloc_traits;
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
        if (ec.value() == ERROR_NETNAME_DELETED)
        {
          if (o->cancel_token_.expired())
            ec = asio::error::operation_aborted;
          else
            ec = asio::error::connection_reset;
        }
        else if (ec.value() == ERROR_PORT_UNREACHABLE)
        {
          ec = asio::error::connection_refused;
        }

        // Check for connection closed.
        else if (!ec && bytes_transferred == 0
            && (o->state_ & socket_ops::stream_oriented) != 0
            && !buffer_sequence_adapter<asio::mutable_buffer,
                MutableBufferSequence>::all_empty(o->buffers_)
            && !boost::is_same<MutableBufferSequence, null_buffers>::value)
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
    socket_ops::state_type state_;
    weak_cancel_token_type cancel_token_;
    MutableBufferSequence buffers_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_receive(base_implementation_type& impl,
      const MutableBufferSequence& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef receive_op<MutableBufferSequence, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, impl.state_,
        impl.cancel_token_, buffers, handler);

    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(buffers);

    start_receive_op(impl, bufs.buffers(), bufs.count(), flags,
        (impl.state_ & socket_ops::stream_oriented) != 0 && bufs.all_empty(),
        ptr.get());
    ptr.release();
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive(base_implementation_type& impl,
      const null_buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    if ((impl.state_ & socket_ops::stream_oriented) != 0)
    {
      // For stream sockets on Windows, we may issue a 0-byte overlapped
      // WSARecv to wait until there is data available on the socket.

      // Allocate and construct an operation to wrap the handler.
      typedef receive_op<null_buffers, Handler> value_type;
      typedef handler_alloc_traits<Handler, value_type> alloc_traits;
      raw_handler_ptr<alloc_traits> raw_ptr(handler);
      handler_ptr<alloc_traits> ptr(raw_ptr, impl.state_,
          impl.cancel_token_, buffers, handler);

      ::WSABUF buf = { 0, 0 };
      start_receive_op(impl, &buf, 1, flags, false, ptr.get());
      ptr.release();
    }
    else
    {
      // Allocate and construct an operation to wrap the handler.
      typedef null_buffers_op<Handler> op;
      typename op::ptr p = { boost::addressof(handler),
        asio_handler_alloc_helpers::allocate(
          sizeof(op), handler), 0 };
      p.p = new (p.v) op(handler);

      start_reactor_op(impl,
          (flags & socket_base::message_out_of_band)
            ? reactor::except_op : reactor::read_op,
          p.p);
      p.v = p.p = 0;
    }
  }

protected:
  // Open a new socket implementation.
  asio::error_code do_open(base_implementation_type& impl,
      int family, int type, int protocol, asio::error_code& ec)
  {
    if (is_open(impl))
    {
      ec = asio::error::already_open;
      return ec;
    }

    socket_holder sock(socket_ops::socket(family, type, protocol, ec));
    if (sock.get() == invalid_socket)
      return ec;

    HANDLE sock_as_handle = reinterpret_cast<HANDLE>(sock.get());
    if (iocp_service_.register_handle(sock_as_handle, ec))
      return ec;

    impl.socket_ = sock.release();
    switch (type)
    {
    case SOCK_STREAM: impl.state_ = socket_ops::stream_oriented; break;
    case SOCK_DGRAM: impl.state_ = socket_ops::datagram_oriented; break;
    default: impl.state_ = 0; break;
    }
    impl.cancel_token_.reset(static_cast<void*>(0), noop_deleter());
    ec = asio::error_code();
    return ec;
  }

  // Assign a native socket to a socket implementation.
  asio::error_code do_assign(base_implementation_type& impl,
      int type, socket_type native_socket, asio::error_code& ec)
  {
    if (is_open(impl))
    {
      ec = asio::error::already_open;
      return ec;
    }

    HANDLE sock_as_handle = reinterpret_cast<HANDLE>(native_socket);
    if (iocp_service_.register_handle(sock_as_handle, ec))
      return ec;

    impl.socket_ = native_socket;
    switch (type)
    {
    case SOCK_STREAM: impl.state_ = socket_ops::stream_oriented; break;
    case SOCK_DGRAM: impl.state_ = socket_ops::datagram_oriented; break;
    default: impl.state_ = 0; break;
    }
    impl.cancel_token_.reset(static_cast<void*>(0), noop_deleter());
    ec = asio::error_code();
    return ec;
  }

  // Helper function to start an asynchronous send operation.
  void start_send_op(base_implementation_type& impl, WSABUF* buffers,
      std::size_t buffer_count, socket_base::message_flags flags,
      bool noop, operation* op)
  {
    update_cancellation_thread_id(impl);
    iocp_service_.work_started();

    if (noop)
      iocp_service_.on_completion(op);
    else if (!is_open(impl))
      iocp_service_.on_completion(op, asio::error::bad_descriptor);
    else
    {
      DWORD bytes_transferred = 0;
      int result = ::WSASend(impl.socket_, buffers,
          buffer_count, &bytes_transferred, flags, op, 0);
      DWORD last_error = ::WSAGetLastError();
      if (last_error == ERROR_PORT_UNREACHABLE)
        last_error = WSAECONNREFUSED;
      if (result != 0 && last_error != WSA_IO_PENDING)
        iocp_service_.on_completion(op, last_error, bytes_transferred);
      else
        iocp_service_.on_pending(op);
    }
  }

  // Helper function to start an asynchronous send_to operation.
  void start_send_to_op(base_implementation_type& impl, WSABUF* buffers,
      std::size_t buffer_count, const socket_addr_type* addr,
      int addrlen, socket_base::message_flags flags, operation* op)
  {
    update_cancellation_thread_id(impl);
    iocp_service_.work_started();

    if (!is_open(impl))
      iocp_service_.on_completion(op, asio::error::bad_descriptor);
    else
    {
      DWORD bytes_transferred = 0;
      int result = ::WSASendTo(impl.socket_, buffers, buffer_count,
          &bytes_transferred, flags, addr, addrlen, op, 0);
      DWORD last_error = ::WSAGetLastError();
      if (last_error == ERROR_PORT_UNREACHABLE)
        last_error = WSAECONNREFUSED;
      if (result != 0 && last_error != WSA_IO_PENDING)
        iocp_service_.on_completion(op, last_error, bytes_transferred);
      else
        iocp_service_.on_pending(op);
    }
  }

  // Helper function to start an asynchronous receive operation.
  void start_receive_op(base_implementation_type& impl, WSABUF* buffers,
      std::size_t buffer_count, socket_base::message_flags flags,
      bool noop, operation* op)
  {
    update_cancellation_thread_id(impl);
    iocp_service_.work_started();

    if (noop)
      iocp_service_.on_completion(op);
    else if (!is_open(impl))
      iocp_service_.on_completion(op, asio::error::bad_descriptor);
    else
    {
      DWORD bytes_transferred = 0;
      DWORD recv_flags = flags;
      int result = ::WSARecv(impl.socket_, buffers, buffer_count,
          &bytes_transferred, &recv_flags, op, 0);
      DWORD last_error = ::WSAGetLastError();
      if (last_error == ERROR_NETNAME_DELETED)
        last_error = WSAECONNRESET;
      else if (last_error == ERROR_PORT_UNREACHABLE)
        last_error = WSAECONNREFUSED;
      if (result != 0 && last_error != WSA_IO_PENDING)
        iocp_service_.on_completion(op, last_error, bytes_transferred);
      else
        iocp_service_.on_pending(op);
    }
  }

  // Helper function to start an asynchronous receive_from operation.
  void start_receive_from_op(base_implementation_type& impl, WSABUF* buffers,
      std::size_t buffer_count, socket_addr_type* addr,
      socket_base::message_flags flags, int* addrlen, operation* op)
  {
    update_cancellation_thread_id(impl);
    iocp_service_.work_started();

    if (!is_open(impl))
      iocp_service_.on_completion(op, asio::error::bad_descriptor);
    else
    {
      DWORD bytes_transferred = 0;
      DWORD recv_flags = flags;
      int result = ::WSARecvFrom(impl.socket_, buffers, buffer_count,
          &bytes_transferred, &recv_flags, addr, addrlen, op, 0);
      DWORD last_error = ::WSAGetLastError();
      if (last_error == ERROR_PORT_UNREACHABLE)
        last_error = WSAECONNREFUSED;
      if (result != 0 && last_error != WSA_IO_PENDING)
        iocp_service_.on_completion(op, last_error, bytes_transferred);
      else
        iocp_service_.on_pending(op);
    }
  }

  // Helper function to start an asynchronous receive_from operation.
  void start_accept_op(base_implementation_type& impl,
      bool peer_is_open, socket_holder& new_socket, int family, int type,
      int protocol, void* output_buffer, DWORD address_length, operation* op)
  {
    update_cancellation_thread_id(impl);
    iocp_service_.work_started();

    if (!is_open(impl))
      iocp_service_.on_completion(op, asio::error::bad_descriptor);
    else if (peer_is_open)
      iocp_service_.on_completion(op, asio::error::already_open);
    else
    {
      asio::error_code ec;
      new_socket.reset(socket_ops::socket(family, type, protocol, ec));
      if (new_socket.get() == invalid_socket)
        iocp_service_.on_completion(op, ec);
      else
      {
        DWORD bytes_read = 0;
        BOOL result = ::AcceptEx(impl.socket_, new_socket.get(), output_buffer,
            0, address_length, address_length, &bytes_read, op);
        DWORD last_error = ::WSAGetLastError();
        if (!result && last_error != WSA_IO_PENDING)
          iocp_service_.on_completion(op, last_error);
        else
          iocp_service_.on_pending(op);
      }
    }
  }

  // Start an asynchronous read or write operation using the the reactor.
  void start_reactor_op(base_implementation_type& impl,
      int op_type, reactor_op* op)
  {
    reactor& r = get_reactor();
    update_cancellation_thread_id(impl);

    if (is_open(impl))
    {
      r.start_op(op_type, impl.socket_, impl.reactor_data_, op, false);
      return;
    }
    else
      op->ec_ = asio::error::bad_descriptor;

    iocp_service_.post_immediate_completion(op);
  }

  // Start the asynchronous connect operation using the reactor.
  void start_connect_op(base_implementation_type& impl,
      reactor_op* op, const socket_addr_type* addr, std::size_t addrlen)
  {
    reactor& r = get_reactor();
    update_cancellation_thread_id(impl);

    if ((impl.state_ & socket_ops::non_blocking) != 0
        || socket_ops::set_internal_non_blocking(
          impl.socket_, impl.state_, op->ec_))
    {
      if (socket_ops::connect(impl.socket_, addr, addrlen, op->ec_) != 0)
      {
        if (op->ec_ == asio::error::in_progress
            || op->ec_ == asio::error::would_block)
        {
          op->ec_ = asio::error_code();
          r.start_op(reactor::connect_op, impl.socket_,
              impl.reactor_data_, op, false);
          return;
        }
      }
    }

    r.post_immediate_completion(op);
  }

  // Helper function to close a socket when the associated object is being
  // destroyed.
  void close_for_destruction(base_implementation_type& impl)
  {
    if (is_open(impl))
    {
      // Check if the reactor was created, in which case we need to close the
      // socket on the reactor as well to cancel any operations that might be
      // running there.
      reactor* r = static_cast<reactor*>(
            interlocked_compare_exchange_pointer(
              reinterpret_cast<void**>(&reactor_), 0, 0));
      if (r)
        r->close_descriptor(impl.socket_, impl.reactor_data_);
    }

    asio::error_code ignored_ec;
    socket_ops::close(impl.socket_, impl.state_, true, ignored_ec);
    impl.socket_ = invalid_socket;
    impl.state_ = 0;
    impl.cancel_token_.reset();
#if defined(ASIO_ENABLE_CANCELIO)
    impl.safe_cancellation_thread_id_ = 0;
#endif // defined(ASIO_ENABLE_CANCELIO)
  }

  // Update the ID of the thread from which cancellation is safe.
  void update_cancellation_thread_id(base_implementation_type& impl)
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

  // Helper function to get the reactor. If no reactor has been created yet, a
  // new one is obtained from the io_service and a pointer to it is cached in
  // this service.
  reactor& get_reactor()
  {
    reactor* r = static_cast<reactor*>(
          interlocked_compare_exchange_pointer(
            reinterpret_cast<void**>(&reactor_), 0, 0));
    if (!r)
    {
      r = &(use_service<reactor>(io_service_));
      interlocked_exchange_pointer(reinterpret_cast<void**>(&reactor_), r);
    }
    return *r;
  }

  // Helper function to emulate InterlockedCompareExchangePointer functionality
  // for:
  // - very old Platform SDKs; and
  // - platform SDKs where MSVC's /Wp64 option causes spurious warnings.
  void* interlocked_compare_exchange_pointer(void** dest, void* exch, void* cmp)
  {
#if defined(_M_IX86)
    return reinterpret_cast<void*>(InterlockedCompareExchange(
          reinterpret_cast<PLONG>(dest), reinterpret_cast<LONG>(exch),
          reinterpret_cast<LONG>(cmp)));
#else
    return InterlockedCompareExchangePointer(dest, exch, cmp);
#endif
  }

  // Helper function to emulate InterlockedExchangePointer functionality for:
  // - very old Platform SDKs; and
  // - platform SDKs where MSVC's /Wp64 option causes spurious warnings.
  void* interlocked_exchange_pointer(void** dest, void* val)
  {
#if defined(_M_IX86)
    return reinterpret_cast<void*>(InterlockedExchange(
          reinterpret_cast<PLONG>(dest), reinterpret_cast<LONG>(val)));
#else
    return InterlockedExchangePointer(dest, val);
#endif
  }

  // The io_service used to obtain the reactor, if required.
  asio::io_service& io_service_;

  // The IOCP service used for running asynchronous operations and dispatching
  // handlers.
  win_iocp_io_service& iocp_service_;

  // The reactor used for performing connect operations. This object is created
  // only if needed.
  reactor* reactor_;

  // Mutex to protect access to the linked list of implementations. 
  asio::detail::mutex mutex_;

  // The head of a linked list of all implementations.
  base_implementation_type* impl_list_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_BASE_HPP
