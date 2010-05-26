//
// detail/win_iocp_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include <cstring>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/addressof.hpp>
#include <boost/weak_ptr.hpp>
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
#include "asio/detail/socket_connect_op.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_iocp_io_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol>
class win_iocp_socket_service
{
public:
  // The protocol type.
  typedef Protocol protocol_type;

  // The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  struct noop_deleter { void operator()(void*) {} };
  typedef boost::shared_ptr<void> shared_cancel_token_type;
  typedef boost::weak_ptr<void> weak_cancel_token_type;

  // The native type of a socket.
  class native_type
  {
  public:
    native_type(socket_type s)
      : socket_(s),
        have_remote_endpoint_(false)
    {
    }

    native_type(socket_type s, const endpoint_type& ep)
      : socket_(s),
        have_remote_endpoint_(true),
        remote_endpoint_(ep)
    {
    }

    void operator=(socket_type s)
    {
      socket_ = s;
      have_remote_endpoint_ = false;
      remote_endpoint_ = endpoint_type();
    }

    operator socket_type() const
    {
      return socket_;
    }

    HANDLE as_handle() const
    {
      return reinterpret_cast<HANDLE>(socket_);
    }

    bool have_remote_endpoint() const
    {
      return have_remote_endpoint_;
    }

    endpoint_type remote_endpoint() const
    {
      return remote_endpoint_;
    }

  private:
    socket_type socket_;
    bool have_remote_endpoint_;
    endpoint_type remote_endpoint_;
  };

  // The implementation type of the socket.
  class implementation_type
  {
  public:
    // Default constructor.
    implementation_type()
      : socket_(invalid_socket),
        state_(0),
        cancel_token_(),
        protocol_(endpoint_type().protocol()),
        next_(0),
        prev_(0)
    {
    }

  private:
    // Only this service will have access to the internal values.
    friend class win_iocp_socket_service;

    // The native socket representation.
    native_type socket_;

    // The current state of the socket.
    socket_ops::state_type state_;

    // We use a shared pointer as a cancellation token here to work around the
    // broken Windows support for cancellation. MSDN says that when you call
    // closesocket any outstanding WSARecv or WSASend operations will complete
    // with the error ERROR_OPERATION_ABORTED. In practice they complete with
    // ERROR_NETNAME_DELETED, which means you can't tell the difference between
    // a local cancellation and the socket being hard-closed by the peer.
    shared_cancel_token_type cancel_token_;

    // The protocol associated with the socket.
    protocol_type protocol_;

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
    implementation_type* next_;
    implementation_type* prev_;
  };

  // Constructor.
  win_iocp_socket_service(asio::io_service& io_service)
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
    implementation_type* impl = impl_list_;
    while (impl)
    {
      asio::error_code ignored_ec;
      close_for_destruction(*impl);
      impl = impl->next_;
    }
  }

  // Construct a new socket implementation.
  void construct(implementation_type& impl)
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

  // Open a new socket implementation.
  asio::error_code open(implementation_type& impl,
      const protocol_type& protocol, asio::error_code& ec)
  {
    if (is_open(impl))
    {
      ec = asio::error::already_open;
      return ec;
    }

    socket_holder sock(socket_ops::socket(protocol.family(), protocol.type(),
          protocol.protocol(), ec));
    if (sock.get() == invalid_socket)
      return ec;

    HANDLE sock_as_handle = reinterpret_cast<HANDLE>(sock.get());
    if (iocp_service_.register_handle(sock_as_handle, ec))
      return ec;

    impl.socket_ = sock.release();
    switch (protocol.type())
    {
    case SOCK_STREAM: impl.state_ = socket_ops::stream_oriented; break;
    case SOCK_DGRAM: impl.state_ = socket_ops::datagram_oriented; break;
    default: impl.state_ = 0; break;
    }
    impl.cancel_token_.reset(static_cast<void*>(0), noop_deleter());
    impl.protocol_ = protocol;
    ec = asio::error_code();
    return ec;
  }

  // Assign a native socket to a socket implementation.
  asio::error_code assign(implementation_type& impl,
      const protocol_type& protocol, const native_type& native_socket,
      asio::error_code& ec)
  {
    if (is_open(impl))
    {
      ec = asio::error::already_open;
      return ec;
    }

    if (iocp_service_.register_handle(native_socket.as_handle(), ec))
      return ec;

    impl.socket_ = native_socket;
    switch (protocol.type())
    {
    case SOCK_STREAM: impl.state_ = socket_ops::stream_oriented; break;
    case SOCK_DGRAM: impl.state_ = socket_ops::datagram_oriented; break;
    default: impl.state_ = 0; break;
    }
    impl.cancel_token_.reset(static_cast<void*>(0), noop_deleter());
    impl.protocol_ = protocol;
    ec = asio::error_code();
    return ec;
  }

  // Determine whether the socket is open.
  bool is_open(const implementation_type& impl) const
  {
    return impl.socket_ != invalid_socket;
  }

  // Destroy a socket implementation.
  asio::error_code close(implementation_type& impl,
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

  // Get the native socket representation.
  native_type native(implementation_type& impl)
  {
    return impl.socket_;
  }

  // Cancel all operations associated with the socket.
  asio::error_code cancel(implementation_type& impl,
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
  bool at_mark(const implementation_type& impl,
      asio::error_code& ec) const
  {
    return socket_ops::sockatmark(impl.socket_, ec);
  }

  // Determine the number of bytes available for reading.
  std::size_t available(const implementation_type& impl,
      asio::error_code& ec) const
  {
    return socket_ops::available(impl.socket_, ec);
  }

  // Bind the socket to the specified local endpoint.
  asio::error_code bind(implementation_type& impl,
      const endpoint_type& endpoint, asio::error_code& ec)
  {
    socket_ops::bind(impl.socket_, endpoint.data(), endpoint.size(), ec);
    return ec;
  }

  // Place the socket into the state where it will listen for new connections.
  asio::error_code listen(implementation_type& impl, int backlog,
      asio::error_code& ec)
  {
    socket_ops::listen(impl.socket_, backlog, ec);
    return ec;
  }

  // Set a socket option.
  template <typename Option>
  asio::error_code set_option(implementation_type& impl,
      const Option& option, asio::error_code& ec)
  {
    socket_ops::setsockopt(impl.socket_, impl.state_,
        option.level(impl.protocol_), option.name(impl.protocol_),
        option.data(impl.protocol_), option.size(impl.protocol_), ec);
    return ec;
  }

  // Set a socket option.
  template <typename Option>
  asio::error_code get_option(const implementation_type& impl,
      Option& option, asio::error_code& ec) const
  {
    std::size_t size = option.size(impl.protocol_);
    socket_ops::getsockopt(impl.socket_, impl.state_,
        option.level(impl.protocol_), option.name(impl.protocol_),
        option.data(impl.protocol_), &size, ec);
    if (!ec)
      option.resize(impl.protocol_, size);
    return ec;
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command>
  asio::error_code io_control(implementation_type& impl,
      IO_Control_Command& command, asio::error_code& ec)
  {
    socket_ops::ioctl(impl.socket_, impl.state_, command.name(),
        static_cast<ioctl_arg_type*>(command.data()), ec);
    return ec;
  }

  // Get the local endpoint.
  endpoint_type local_endpoint(const implementation_type& impl,
      asio::error_code& ec) const
  {
    endpoint_type endpoint;
    std::size_t addr_len = endpoint.capacity();
    if (socket_ops::getsockname(impl.socket_, endpoint.data(), &addr_len, ec))
      return endpoint_type();
    endpoint.resize(addr_len);
    return endpoint;
  }

  // Get the remote endpoint.
  endpoint_type remote_endpoint(const implementation_type& impl,
      asio::error_code& ec) const
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return endpoint_type();
    }

    if (impl.socket_.have_remote_endpoint())
    {
      // Check if socket is still connected.
      DWORD connect_time = 0;
      size_t connect_time_len = sizeof(connect_time);
      if (socket_ops::getsockopt(impl.socket_, impl.state_,
            SOL_SOCKET, SO_CONNECT_TIME, &connect_time,
            &connect_time_len, ec) == socket_error_retval)
      {
        return endpoint_type();
      }
      if (connect_time == 0xFFFFFFFF)
      {
        ec = asio::error::not_connected;
        return endpoint_type();
      }

      ec = asio::error_code();
      return impl.socket_.remote_endpoint();
    }
    else
    {
      endpoint_type endpoint;
      std::size_t addr_len = endpoint.capacity();
      if (socket_ops::getpeername(impl.socket_, endpoint.data(), &addr_len, ec))
        return endpoint_type();
      endpoint.resize(addr_len);
      return endpoint;
    }
  }

  /// Disable sends or receives on the socket.
  asio::error_code shutdown(implementation_type& impl,
      socket_base::shutdown_type what, asio::error_code& ec)
  {
    socket_ops::shutdown(impl.socket_, what, ec);
    return ec;
  }

  // Send the given data to the peer. Returns the number of bytes sent.
  template <typename ConstBufferSequence>
  size_t send(implementation_type& impl, const ConstBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence> bufs(buffers);

    return socket_ops::sync_send(impl.socket_, impl.state_,
        bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
  }

  // Wait until data can be sent without blocking.
  size_t send(implementation_type& impl, const null_buffers&,
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
  void async_send(implementation_type& impl, const ConstBufferSequence& buffers,
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
        impl.protocol_.type() == SOCK_STREAM && bufs.all_empty(), ptr.get());
    ptr.release();
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler>
  void async_send(implementation_type& impl, const null_buffers&,
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

  // Send a datagram to the specified endpoint. Returns the number of bytes
  // sent.
  template <typename ConstBufferSequence>
  size_t send_to(implementation_type& impl, const ConstBufferSequence& buffers,
      const endpoint_type& destination, socket_base::message_flags flags,
      asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence> bufs(buffers);

    return socket_ops::sync_sendto(impl.socket_, impl.state_,
        bufs.buffers(), bufs.count(), flags,
        destination.data(), destination.size(), ec);
  }

  // Wait until data can be sent without blocking.
  size_t send_to(implementation_type& impl, const null_buffers&,
      socket_base::message_flags, const endpoint_type&,
      asio::error_code& ec)
  {
    // Wait for socket to become ready.
    socket_ops::poll_write(impl.socket_, ec);

    return 0;
  }

  template <typename ConstBufferSequence, typename Handler>
  class send_to_op : public operation
  {
  public:
    send_to_op(weak_cancel_token_type cancel_token,
        const ConstBufferSequence& buffers, Handler handler)
      : operation(&send_to_op::do_complete),
        cancel_token_(cancel_token),
        buffers_(buffers),
        handler_(handler)
    {
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code ec, std::size_t bytes_transferred)
    {
      // Take ownership of the operation object.
      send_to_op* o(static_cast<send_to_op*>(base));
      typedef handler_alloc_traits<Handler, send_to_op> alloc_traits;
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
        if (ec.value() == ERROR_PORT_UNREACHABLE)
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
  void async_send_to(implementation_type& impl,
      const ConstBufferSequence& buffers, const endpoint_type& destination,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef send_to_op<ConstBufferSequence, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr,
        impl.cancel_token_, buffers, handler);

    buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence> bufs(buffers);

    start_send_to_op(impl, bufs.buffers(),
        bufs.count(), destination, flags, ptr.get());
    ptr.release();
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler>
  void async_send_to(implementation_type& impl, const null_buffers&,
      socket_base::message_flags, const endpoint_type&, Handler handler)
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
  size_t receive(implementation_type& impl,
      const MutableBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(buffers);

    return socket_ops::sync_recv(impl.socket_, impl.state_,
        bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
  }

  // Wait until data can be received without blocking.
  size_t receive(implementation_type& impl, const null_buffers&,
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
    receive_op(int protocol_type, weak_cancel_token_type cancel_token,
        const MutableBufferSequence& buffers, Handler handler)
      : operation(&receive_op::do_complete),
        protocol_type_(protocol_type),
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
            && o->protocol_type_ == SOCK_STREAM
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
    int protocol_type_;
    weak_cancel_token_type cancel_token_;
    MutableBufferSequence buffers_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_receive(implementation_type& impl,
      const MutableBufferSequence& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef receive_op<MutableBufferSequence, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    int protocol_type = impl.protocol_.type();
    handler_ptr<alloc_traits> ptr(raw_ptr, protocol_type,
        impl.cancel_token_, buffers, handler);

    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(buffers);

    start_receive_op(impl, bufs.buffers(), bufs.count(), flags,
        protocol_type == SOCK_STREAM && bufs.all_empty(), ptr.get());
    ptr.release();
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive(implementation_type& impl, const null_buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    if (impl.protocol_.type() == SOCK_STREAM)
    {
      // For stream sockets on Windows, we may issue a 0-byte overlapped
      // WSARecv to wait until there is data available on the socket.

      // Allocate and construct an operation to wrap the handler.
      typedef receive_op<null_buffers, Handler> value_type;
      typedef handler_alloc_traits<Handler, value_type> alloc_traits;
      raw_handler_ptr<alloc_traits> raw_ptr(handler);
      int protocol_type = impl.protocol_.type();
      handler_ptr<alloc_traits> ptr(raw_ptr, protocol_type,
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

  // Receive a datagram with the endpoint of the sender. Returns the number of
  // bytes received.
  template <typename MutableBufferSequence>
  size_t receive_from(implementation_type& impl,
      const MutableBufferSequence& buffers,
      endpoint_type& sender_endpoint, socket_base::message_flags flags,
      asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(buffers);

    std::size_t addr_len = sender_endpoint.capacity();
    std::size_t bytes_recvd = socket_ops::sync_recvfrom(
        impl.socket_, impl.state_, bufs.buffers(), bufs.count(),
        flags, sender_endpoint.data(), &addr_len, ec);

    if (!ec)
      sender_endpoint.resize(addr_len);

    return bytes_recvd;
  }

  // Wait until data can be received without blocking.
  size_t receive_from(implementation_type& impl,
      const null_buffers&, endpoint_type& sender_endpoint,
      socket_base::message_flags, asio::error_code& ec)
  {
    // Wait for socket to become ready.
    socket_ops::poll_read(impl.socket_, ec);

    // Reset endpoint since it can be given no sensible value at this time.
    sender_endpoint = endpoint_type();

    return 0;
  }

  template <typename MutableBufferSequence, typename Handler>
  class receive_from_op : public operation
  {
  public:
    receive_from_op(int protocol_type, endpoint_type& endpoint,
        const MutableBufferSequence& buffers, Handler handler)
      : operation(&receive_from_op::do_complete),
        protocol_type_(protocol_type),
        endpoint_(endpoint),
        endpoint_size_(static_cast<int>(endpoint.capacity())),
        buffers_(buffers),
        handler_(handler)
    {
    }

    int& endpoint_size()
    {
      return endpoint_size_;
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code ec, std::size_t bytes_transferred)
    {
      // Take ownership of the operation object.
      receive_from_op* o(static_cast<receive_from_op*>(base));
      typedef handler_alloc_traits<Handler, receive_from_op> alloc_traits;
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
        if (ec.value() == ERROR_PORT_UNREACHABLE)
        {
          ec = asio::error::connection_refused;
        }

        // Record the size of the endpoint returned by the operation.
        o->endpoint_.resize(o->endpoint_size_);

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
    int protocol_type_;
    endpoint_type& endpoint_;
    int endpoint_size_;
    weak_cancel_token_type cancel_token_;
    MutableBufferSequence buffers_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_endpoint object must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_receive_from(implementation_type& impl,
      const MutableBufferSequence& buffers, endpoint_type& sender_endp,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef receive_from_op<MutableBufferSequence, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    int protocol_type = impl.protocol_.type();
    handler_ptr<alloc_traits> ptr(raw_ptr,
        protocol_type, sender_endp, buffers, handler);

    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(buffers);

    start_receive_from_op(impl, bufs.buffers(), bufs.count(),
        sender_endp, flags, &ptr.get()->endpoint_size(), ptr.get());
    ptr.release();
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive_from(implementation_type& impl,
      const null_buffers&, endpoint_type& sender_endpoint,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef null_buffers_op<Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(handler);

    // Reset endpoint since it can be given no sensible value at this time.
    sender_endpoint = endpoint_type();

    start_reactor_op(impl,
        (flags & socket_base::message_out_of_band)
          ? reactor::except_op : reactor::read_op,
        p.p);
    p.v = p.p = 0;
  }

  // Accept a new connection.
  template <typename Socket>
  asio::error_code accept(implementation_type& impl, Socket& peer,
      endpoint_type* peer_endpoint, asio::error_code& ec)
  {
    // We cannot accept a socket that is already open.
    if (peer.is_open())
    {
      ec = asio::error::already_open;
      return ec;
    }

    std::size_t addr_len = peer_endpoint ? peer_endpoint->capacity() : 0;
    socket_holder new_socket(socket_ops::sync_accept(impl.socket_,
          impl.state_, peer_endpoint ? peer_endpoint->data() : 0,
          peer_endpoint ? &addr_len : 0, ec));

    // On success, assign new connection to peer socket object.
    if (new_socket.get() >= 0)
    {
      if (peer_endpoint)
        peer_endpoint->resize(addr_len);
      if (!peer.assign(impl.protocol_, new_socket.get(), ec))
        new_socket.release();
    }

    return ec;
  }

  template <typename Socket, typename Handler>
  class accept_op : public operation
  {
  public:
    accept_op(win_iocp_io_service& iocp_service, socket_type socket,
        Socket& peer, const protocol_type& protocol,
        endpoint_type* peer_endpoint, bool enable_connection_aborted,
        Handler handler)
      : operation(&accept_op::do_complete),
        iocp_service_(iocp_service),
        socket_(socket),
        peer_(peer),
        protocol_(protocol),
        peer_endpoint_(peer_endpoint),
        enable_connection_aborted_(enable_connection_aborted),
        handler_(handler)
    {
    }

    socket_holder& new_socket()
    {
      return new_socket_;
    }

    void* output_buffer()
    {
      return output_buffer_;
    }

    DWORD address_length()
    {
      return sizeof(sockaddr_storage_type) + 16;
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code ec, std::size_t /*bytes_transferred*/)
    {
      // Take ownership of the handler object.
      accept_op* o(static_cast<accept_op*>(base));
      typedef handler_alloc_traits<Handler, accept_op> alloc_traits;
      handler_ptr<alloc_traits> ptr(o->handler_, o);

      // Make the upcall if required.
      if (owner)
      {
        // Map Windows error ERROR_NETNAME_DELETED to connection_aborted.
        if (ec.value() == ERROR_NETNAME_DELETED)
        {
          ec = asio::error::connection_aborted;
        }

        // Restart the accept operation if we got the connection_aborted error
        // and the enable_connection_aborted socket option is not set.
        if (ec == asio::error::connection_aborted
            && !o->enable_connection_aborted_)
        {
          // Reset OVERLAPPED structure.
          o->reset();

          // Create a new socket for the next connection, since the AcceptEx
          // call fails with WSAEINVAL if we try to reuse the same socket.
          o->new_socket_.reset();
          o->new_socket_.reset(socket_ops::socket(o->protocol_.family(),
                o->protocol_.type(), o->protocol_.protocol(), ec));
          if (o->new_socket_.get() != invalid_socket)
          {
            // Accept a connection.
            DWORD bytes_read = 0;
            BOOL result = ::AcceptEx(o->socket_, o->new_socket_.get(),
                o->output_buffer(), 0, o->address_length(),
                o->address_length(), &bytes_read, o);
            DWORD last_error = ::WSAGetLastError();
            ec = asio::error_code(last_error,
                asio::error::get_system_category());

            // Check if the operation completed immediately.
            if (!result && last_error != WSA_IO_PENDING)
            {
              if (last_error == ERROR_NETNAME_DELETED
                  || last_error == WSAECONNABORTED)
              {
                // Post this handler so that operation will be restarted again.
                o->iocp_service_.work_started();
                o->iocp_service_.on_completion(o, ec);
                ptr.release();
                return;
              }
              else
              {
                // Operation already complete. Continue with rest of this
                // handler.
              }
            }
            else
            {
              // Asynchronous operation has been successfully restarted.
              o->iocp_service_.work_started();
              o->iocp_service_.on_pending(o);
              ptr.release();
              return;
            }
          }
        }

        // Get the address of the peer.
        endpoint_type peer_endpoint;
        if (!ec)
        {
          LPSOCKADDR local_addr = 0;
          int local_addr_length = 0;
          LPSOCKADDR remote_addr = 0;
          int remote_addr_length = 0;
          GetAcceptExSockaddrs(o->output_buffer(), 0, o->address_length(),
              o->address_length(), &local_addr, &local_addr_length,
              &remote_addr, &remote_addr_length);
          if (static_cast<std::size_t>(remote_addr_length)
              > peer_endpoint.capacity())
          {
            ec = asio::error::invalid_argument;
          }
          else
          {
            using namespace std; // For memcpy.
            memcpy(peer_endpoint.data(), remote_addr, remote_addr_length);
            peer_endpoint.resize(static_cast<std::size_t>(remote_addr_length));
          }
        }

        // Need to set the SO_UPDATE_ACCEPT_CONTEXT option so that getsockname
        // and getpeername will work on the accepted socket.
        if (!ec)
        {
          SOCKET update_ctx_param = o->socket_;
          socket_ops::state_type state = 0;
          socket_ops::setsockopt(o->new_socket_.get(), state,
                SOL_SOCKET, SO_UPDATE_ACCEPT_CONTEXT,
                &update_ctx_param, sizeof(SOCKET), ec);
        }

        // If the socket was successfully accepted, transfer ownership of the
        // socket to the peer object.
        if (!ec)
        {
          o->peer_.assign(o->protocol_,
              native_type(o->new_socket_.get(), peer_endpoint), ec);
          if (!ec)
            o->new_socket_.release();
        }

        // Pass endpoint back to caller.
        if (o->peer_endpoint_)
          *o->peer_endpoint_ = peer_endpoint;

        // Make a copy of the handler so that the memory can be deallocated
        // before the upcall is made. Even if we're not about to make an
        // upcall, a sub-object of the handler may be the true owner of the
        // memory associated with the handler. Consequently, a local copy of
        // the handler is required to ensure that any owning sub-object remains
        // valid until after we have deallocated the memory here.
        detail::binder1<Handler, asio::error_code>
          handler(o->handler_, ec);
        ptr.reset();
        asio::detail::fenced_block b;
        asio_handler_invoke_helpers::invoke(handler, handler);
      }
    }

  private:
    win_iocp_io_service& iocp_service_;
    socket_type socket_;
    socket_holder new_socket_;
    Socket& peer_;
    protocol_type protocol_;
    endpoint_type* peer_endpoint_;
    unsigned char output_buffer_[(sizeof(sockaddr_storage_type) + 16) * 2];
    bool enable_connection_aborted_;
    Handler handler_;
  };

  // Start an asynchronous accept. The peer and peer_endpoint objects
  // must be valid until the accept's handler is invoked.
  template <typename Socket, typename Handler>
  void async_accept(implementation_type& impl, Socket& peer,
      endpoint_type* peer_endpoint, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef accept_op<Socket, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    bool enable_connection_aborted =
      (impl.state_ & socket_ops::enable_connection_aborted);
    handler_ptr<alloc_traits> ptr(raw_ptr, iocp_service_, impl.socket_, peer,
        impl.protocol_, peer_endpoint, enable_connection_aborted, handler);

    start_accept_op(impl, peer.is_open(), ptr.get()->new_socket(),
        ptr.get()->output_buffer(), ptr.get()->address_length(), ptr.get());
    ptr.release();
  }

  // Connect the socket to the specified endpoint.
  asio::error_code connect(implementation_type& impl,
      const endpoint_type& peer_endpoint, asio::error_code& ec)
  {
    socket_ops::sync_connect(impl.socket_,
        peer_endpoint.data(), peer_endpoint.size(), ec);
    return ec;
  }

  // Start an asynchronous connect.
  template <typename Handler>
  void async_connect(implementation_type& impl,
      const endpoint_type& peer_endpoint, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef socket_connect_op<Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(impl.socket_, handler);

    start_connect_op(impl, p.p, peer_endpoint);
    p.v = p.p = 0;
  }

private:
  // Helper function to start an asynchronous send operation.
  void start_send_op(implementation_type& impl, WSABUF* buffers,
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
  void start_send_to_op(implementation_type& impl, WSABUF* buffers,
      std::size_t buffer_count, const endpoint_type& destination,
      socket_base::message_flags flags, operation* op)
  {
    update_cancellation_thread_id(impl);
    iocp_service_.work_started();

    if (!is_open(impl))
      iocp_service_.on_completion(op, asio::error::bad_descriptor);
    else
    {
      DWORD bytes_transferred = 0;
      int result = ::WSASendTo(impl.socket_, buffers, buffer_count,
          &bytes_transferred, flags, destination.data(),
          static_cast<int>(destination.size()), op, 0);
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
  void start_receive_op(implementation_type& impl, WSABUF* buffers,
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
  void start_receive_from_op(implementation_type& impl, WSABUF* buffers,
      std::size_t buffer_count, endpoint_type& sender_endpoint,
      socket_base::message_flags flags, int* endpoint_size, operation* op)
  {
    update_cancellation_thread_id(impl);
    iocp_service_.work_started();

    if (!is_open(impl))
      iocp_service_.on_completion(op, asio::error::bad_descriptor);
    else
    {
      DWORD bytes_transferred = 0;
      DWORD recv_flags = flags;
      int result = ::WSARecvFrom(impl.socket_, buffers,
          buffer_count, &bytes_transferred, &recv_flags,
          sender_endpoint.data(), endpoint_size, op, 0);
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
  void start_accept_op(implementation_type& impl,
      bool peer_is_open, socket_holder& new_socket,
      void* output_buffer, DWORD address_length, operation* op)
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
      new_socket.reset(socket_ops::socket(impl.protocol_.family(),
            impl.protocol_.type(), impl.protocol_.protocol(), ec));
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
  void start_reactor_op(implementation_type& impl, int op_type, reactor_op* op)
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
  void start_connect_op(implementation_type& impl,
      reactor_op* op, const endpoint_type& peer_endpoint)
  {
    reactor& r = get_reactor();
    update_cancellation_thread_id(impl);

    if ((impl.state_ & socket_ops::non_blocking)
        || socket_ops::set_internal_non_blocking(
          impl.socket_, impl.state_, op->ec_))
    {
      if (socket_ops::connect(impl.socket_, peer_endpoint.data(),
            peer_endpoint.size(), op->ec_) != 0)
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
  void close_for_destruction(implementation_type& impl)
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
  implementation_type* impl_list_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP
