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
#include "asio/detail/win_iocp_null_buffers_op.hpp"
#include "asio/detail/win_iocp_socket_send_op.hpp"
#include "asio/detail/win_iocp_socket_service_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol>
class win_iocp_socket_service : public win_iocp_socket_service_base
{
public:
  // The protocol type.
  typedef Protocol protocol_type;

  // The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  struct noop_deleter { void operator()(void*) {} };
  typedef shared_ptr<void> shared_cancel_token_type;
  typedef weak_ptr<void> weak_cancel_token_type;

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
  struct implementation_type :
    win_iocp_socket_service_base::base_implementation_type
  {
    // Default constructor.
    implementation_type()
      : protocol_(endpoint_type().protocol()),
        have_remote_endpoint_(false),
        remote_endpoint_()
    {
    }

    // The protocol associated with the socket.
    protocol_type protocol_;

    // Whether we have a cached remote endpoint.
    bool have_remote_endpoint_;

    // A cached remote endpoint.
    endpoint_type remote_endpoint_;
  };

  // Constructor.
  win_iocp_socket_service(asio::io_service& io_service)
    : win_iocp_socket_service_base(io_service)
  {
  }

  // Open a new socket implementation.
  asio::error_code open(implementation_type& impl,
      const protocol_type& protocol, asio::error_code& ec)
  {
    if (!do_open(impl, protocol.family(),
          protocol.type(), protocol.protocol(), ec))
    {
      impl.protocol_ = protocol;
      impl.have_remote_endpoint_ = false;
      impl.remote_endpoint_ = endpoint_type();
    }
    return ec;
  }

  // Assign a native socket to a socket implementation.
  asio::error_code assign(implementation_type& impl,
      const protocol_type& protocol, const native_type& native_socket,
      asio::error_code& ec)
  {
    if (!do_assign(impl, protocol.type(), native_socket, ec))
    {
      impl.protocol_ = protocol;
      impl.have_remote_endpoint_ = native_socket.have_remote_endpoint();
      impl.remote_endpoint_ = native_socket.remote_endpoint();
    }
    return ec;
  }

  // Get the native socket representation.
  native_type native(implementation_type& impl)
  {
    if (impl.have_remote_endpoint_)
      return native_type(impl.socket_, impl.remote_endpoint_);
    return native_type(impl.socket_);
  }

  // Bind the socket to the specified local endpoint.
  asio::error_code bind(implementation_type& impl,
      const endpoint_type& endpoint, asio::error_code& ec)
  {
    socket_ops::bind(impl.socket_, endpoint.data(), endpoint.size(), ec);
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

    if (impl.have_remote_endpoint_)
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
      return impl.remote_endpoint_;
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

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_send_to(implementation_type& impl,
      const ConstBufferSequence& buffers, const endpoint_type& destination,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef win_iocp_socket_send_op<ConstBufferSequence, Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(impl.cancel_token_, buffers, handler);

    buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence> bufs(buffers);

    start_send_to_op(impl, bufs.buffers(), bufs.count(),
        destination.data(), static_cast<int>(destination.size()),
        flags, p.p);
    p.v = p.p = 0;
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler>
  void async_send_to(implementation_type& impl, const null_buffers&,
      socket_base::message_flags, const endpoint_type&, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef win_iocp_null_buffers_op<Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(impl.cancel_token_, handler);

    start_reactor_op(impl, reactor::write_op, p.p);
    p.v = p.p = 0;
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
        sender_endp.data(), flags, &ptr.get()->endpoint_size(), ptr.get());
    ptr.release();
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive_from(implementation_type& impl,
      const null_buffers&, endpoint_type& sender_endpoint,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef win_iocp_null_buffers_op<Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(impl.cancel_token_, handler);

    // Reset endpoint since it can be given no sensible value at this time.
    sender_endpoint = endpoint_type();

    start_null_buffers_receive_op(impl, flags, p.p);
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
      (impl.state_ & socket_ops::enable_connection_aborted) != 0;
    handler_ptr<alloc_traits> ptr(raw_ptr, iocp_service_, impl.socket_, peer,
        impl.protocol_, peer_endpoint, enable_connection_aborted, handler);

    start_accept_op(impl, peer.is_open(), ptr.get()->new_socket(),
        impl.protocol_.family(), impl.protocol_.type(),
        impl.protocol_.protocol(), ptr.get()->output_buffer(),
        ptr.get()->address_length(), ptr.get());
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

    start_connect_op(impl, p.p, peer_endpoint.data(),
        static_cast<int>(peer_endpoint.size()));
    p.v = p.p = 0;
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP
