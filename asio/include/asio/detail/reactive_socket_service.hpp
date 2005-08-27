//
// reactive_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <typename Demuxer, typename Reactor>
class reactive_socket_service
{
public:
  // The native type of the socket. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_type impl_type;

  // Constructor.
  reactive_socket_service(Demuxer& d)
    : demuxer_(d),
      reactor_(d.get_service(service_factory<Reactor>()))
  {
  }

  // The demuxer type for this service.
  typedef Demuxer demuxer_type;

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Return a null socket implementation.
  static impl_type null()
  {
    return invalid_socket;
  }

  // Open a new socket implementation.
  template <typename Protocol, typename Error_Handler>
  void open(impl_type& impl, const Protocol& protocol,
      Error_Handler error_handler)
  {
    socket_holder sock(socket_ops::socket(protocol.family(),
          protocol.type(), protocol.protocol()));
    if (sock.get() == invalid_socket)
      error_handler(asio::error(socket_ops::get_error()));
    else
      impl = sock.release();
  }

  // Assign a new socket implementation.
  void assign(impl_type& impl, impl_type new_impl)
  {
    impl = new_impl;
  }

  // Destroy a socket implementation.
  void close(impl_type& impl)
  {
    if (impl != null())
    {
      reactor_.close_descriptor(impl, socket_ops::close);
      impl = null();
    }
  }

  // Bind the socket to the specified local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void bind(impl_type& impl, const Endpoint& endpoint,
      Error_Handler error_handler)
  {
    if (socket_ops::bind(impl, endpoint.native_data(),
          endpoint.native_size()) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Place the socket into the state where it will listen for new connections.
  template <typename Error_Handler>
  void listen(impl_type& impl, int backlog, Error_Handler error_handler)
  {
    if (backlog == 0)
      backlog = SOMAXCONN;

    if (socket_ops::listen(impl, backlog) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void set_option(impl_type& impl, const Option& option,
      Error_Handler error_handler)
  {
    if (socket_ops::setsockopt(impl, option.level(), option.name(),
          option.data(), option.size()))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(const impl_type& impl, Option& option,
      Error_Handler error_handler) const
  {
    size_t size = option.size();
    if (socket_ops::getsockopt(impl, option.level(), option.name(),
          option.data(), &size))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command, typename Error_Handler>
  void io_control(impl_type& impl, IO_Control_Command& command,
      Error_Handler error_handler)
  {
    if (socket_ops::ioctl(impl, command.name(),
          static_cast<ioctl_arg_type*>(command.data())))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(const impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    socket_addr_len_type addr_len = endpoint.native_size();
    if (socket_ops::getsockname(impl, endpoint.native_data(), &addr_len))
      error_handler(asio::error(socket_ops::get_error()));
    endpoint.native_size(addr_len);
  }

  /// Disable sends or receives on the socket.
  template <typename Error_Handler>
  void shutdown(impl_type& impl, socket_base::shutdown_type what,
      Error_Handler error_handler)
  {
    if (socket_ops::shutdown(impl, what) != 0)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Send the given data to the peer. Returns the number of bytes sent or
  // 0 if the connection was closed cleanly.
  template <typename Error_Handler>
  size_t send(impl_type& impl, const void* data, size_t length,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    int bytes_sent = socket_ops::send(impl, data, length, flags);
    if (bytes_sent < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }
    return bytes_sent;
  }

  template <typename Handler>
  class send_handler
  {
  public:
    send_handler(impl_type impl, Demuxer& demuxer, const void* data,
        size_t length, socket_base::message_flags flags, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        length_(length),
        flags_(flags),
        handler_(handler)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::send(impl_, data_, length_, flags_);
      asio::error error(bytes < 0
          ? socket_ops::get_error() : asio::error::success);
      demuxer_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error, 0));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    const void* data_;
    size_t length_;
    socket_base::message_flags flags_;
    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Handler>
  void async_send(impl_type& impl, const void* data, size_t length,
      socket_base::message_flags flags, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error, 0));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_write_op(impl,
          send_handler<Handler>(impl, demuxer_, data, length, flags, handler));
    }
  }

  // Send a datagram to the specified endpoint. Returns the number of bytes
  // sent.
  template <typename Endpoint, typename Error_Handler>
  size_t send_to(impl_type& impl, const void* data, size_t length,
      const Endpoint& destination, Error_Handler error_handler)
  {
    int bytes_sent = socket_ops::sendto(impl, data, length, 0,
        destination.native_data(), destination.native_size());
    if (bytes_sent < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }
    return bytes_sent;
  }

  template <typename Endpoint, typename Handler>
  class send_to_handler
  {
  public:
    send_to_handler(impl_type impl, Demuxer& demuxer, const void* data,
        size_t length, const Endpoint& endpoint, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        length_(length),
        destination_(endpoint),
        handler_(handler)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::sendto(impl_, data_, length_, 0,
          destination_.native_data(), destination_.native_size());
      asio::error error(bytes < 0
          ? socket_ops::get_error() : asio::error::success);
      demuxer_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error, 0));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    const void* data_;
    size_t length_;
    Endpoint destination_;
    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Endpoint, typename Handler>
  void async_send_to(impl_type& impl, const void* data, size_t length,
      const Endpoint& destination, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error, 0));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_write_op(impl, send_to_handler<Endpoint, Handler>(
            impl, demuxer_, data, length, destination, handler));
    }
  }

  // Receive some data from the peer. Returns the number of bytes received or
  // 0 if the connection was closed cleanly.
  template <typename Error_Handler>
  size_t receive(impl_type& impl, void* data, size_t max_length,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    int bytes_recvd = socket_ops::recv(impl, data, max_length, flags);
    if (bytes_recvd < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }
    return bytes_recvd;
  }

  template <typename Handler>
  class receive_handler
  {
  public:
    receive_handler(impl_type impl, Demuxer& demuxer, void* data,
        size_t max_length, socket_base::message_flags flags, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        max_length_(max_length),
        flags_(flags),
        handler_(handler)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::recv(impl_, data_, max_length_, flags_);
      asio::error error(bytes < 0
          ? socket_ops::get_error() : asio::error::success);
      demuxer_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error, 0));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    void* data_;
    size_t max_length_;
    socket_base::message_flags flags_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename Handler>
  void async_receive(impl_type& impl, void* data, size_t max_length,
      socket_base::message_flags flags, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error, 0));
    }
    else
    {
      demuxer_.work_started();
      if (flags & socket_base::message_out_of_band)
      {
        reactor_.start_except_op(impl,
            receive_handler<Handler>(impl, demuxer_,
              data, max_length, flags, handler));
      }
      else
      {
        reactor_.start_read_op(impl,
            receive_handler<Handler>(impl, demuxer_,
              data, max_length, flags, handler));
      }
    }
  }

  // Receive a datagram with the endpoint of the sender. Returns the number of
  // bytes received.
  template <typename Endpoint, typename Error_Handler>
  size_t receive_from(impl_type& impl, void* data, size_t max_length,
      Endpoint& sender_endpoint, Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = sender_endpoint.native_size();
    int bytes_recvd = socket_ops::recvfrom(impl, data, max_length, 0,
        sender_endpoint.native_data(), &addr_len);
    if (bytes_recvd < 0)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return 0;
    }

    sender_endpoint.native_size(addr_len);

    return bytes_recvd;
  }

  template <typename Endpoint, typename Handler>
  class receive_from_handler
  {
  public:
    receive_from_handler(impl_type impl, Demuxer& demuxer, void* data,
        size_t max_length, Endpoint& endpoint, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        max_length_(max_length),
        sender_endpoint_(endpoint),
        handler_(handler)
    {
    }

    void do_operation()
    {
      socket_addr_len_type addr_len = sender_endpoint_.native_size();
      int bytes = socket_ops::recvfrom(impl_, data_, max_length_, 0,
          sender_endpoint_.native_data(), &addr_len);
      asio::error error(bytes < 0
          ? socket_ops::get_error() : asio::error::success);
      sender_endpoint_.native_size(addr_len);
      demuxer_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error, 0));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    void* data_;
    size_t max_length_;
    Endpoint& sender_endpoint_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_endpoint object must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename Endpoint, typename Handler>
  void async_receive_from(impl_type& impl, void* data, size_t max_length,
      Endpoint& sender_endpoint, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error, 0));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_read_op(impl, receive_from_handler<Endpoint, Handler>(
            impl, demuxer_, data, max_length, sender_endpoint, handler));
    }
  }

  // Accept a new connection.
  template <typename Socket, typename Error_Handler>
  void accept(impl_type& impl, Socket& peer, Error_Handler error_handler)
  {
    // We cannot accept a socket that is already open.
    if (peer.impl() != invalid_socket)
    {
      error_handler(asio::error(asio::error::already_connected));
      return;
    }

    socket_type new_socket = socket_ops::accept(impl, 0, 0);
    if (int err = socket_ops::get_error())
    {
      error_handler(asio::error(err));
      return;
    }

    peer.set_impl(new_socket);
  }

  // Accept a new connection.
  template <typename Socket, typename Endpoint, typename Error_Handler>
  void accept_endpoint(impl_type& impl, Socket& peer, Endpoint& peer_endpoint,
      Error_Handler error_handler)
  {
    // We cannot accept a socket that is already open.
    if (peer.impl() != invalid_socket)
    {
      error_handler(asio::error(asio::error::already_connected));
      return;
    }

    socket_addr_len_type addr_len = peer_endpoint.native_size();
    socket_type new_socket = socket_ops::accept(impl,
        peer_endpoint.native_data(), &addr_len);
    if (int err = socket_ops::get_error())
    {
      error_handler(asio::error(err));
      return;
    }

    peer_endpoint.native_size(addr_len);

    peer.set_impl(new_socket);
  }

  template <typename Socket, typename Handler>
  class accept_handler
  {
  public:
    accept_handler(impl_type impl, Demuxer& demuxer, Socket& peer,
        Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        peer_(peer),
        handler_(handler)
    {
    }

    void do_operation()
    {
      socket_type new_socket = socket_ops::accept(impl_, 0, 0);
      asio::error error(new_socket == invalid_socket
          ? socket_ops::get_error() : asio::error::success);
      peer_.set_impl(new_socket);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    Socket& peer_;
    Handler handler_;
  };

  // Start an asynchronous accept. The peer object must be valid until the
  // accept's handler is invoked.
  template <typename Socket, typename Handler>
  void async_accept(impl_type& impl, Socket& peer, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error));
    }
    else if (peer.impl() != invalid_socket)
    {
      asio::error error(asio::error::already_connected);
      demuxer_.post(bind_handler(handler, error));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_read_op(impl,
          accept_handler<Socket, Handler>(impl, demuxer_, peer, handler));
    }
  }

  template <typename Socket, typename Endpoint, typename Handler>
  class accept_endp_handler
  {
  public:
    accept_endp_handler(impl_type impl, Demuxer& demuxer, Socket& peer,
        Endpoint& peer_endpoint, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        peer_(peer),
        peer_endpoint_(peer_endpoint),
        handler_(handler)
    {
    }

    void do_operation()
    {
      socket_addr_len_type addr_len = peer_endpoint_.native_size();
      socket_type new_socket = socket_ops::accept(impl_,
          peer_endpoint_.native_data(), &addr_len);
      asio::error error(new_socket == invalid_socket
          ? socket_ops::get_error() : asio::error::success);
      peer_endpoint_.native_size(addr_len);
      peer_.set_impl(new_socket);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    Socket& peer_;
    Endpoint& peer_endpoint_;
    Handler handler_;
  };

  // Start an asynchronous accept. The peer and peer_endpoint objects
  // must be valid until the accept's handler is invoked.
  template <typename Socket, typename Endpoint, typename Handler>
  void async_accept_endpoint(impl_type& impl, Socket& peer,
      Endpoint& peer_endpoint, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error));
    }
    else if (peer.impl() != invalid_socket)
    {
      asio::error error(asio::error::already_connected);
      demuxer_.post(bind_handler(handler, error));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_read_op(impl,
          accept_endp_handler<Socket, Endpoint, Handler>(
            impl, demuxer_, peer, peer_endpoint, handler));
    }
  }

private:
  // The demuxer used for dispatching handlers.
  Demuxer& demuxer_;

  // The selector that performs event demultiplexing for the provider.
  Reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_HPP
