//
// reactive_socket_acceptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_SOCKET_ACCEPTOR_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_ACCEPTOR_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_stream_socket.hpp"
#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <typename Demuxer, typename Reactor>
class reactive_socket_acceptor_service
{
public:
  // The native type of the socket acceptor. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_type impl_type;

  // Return a null socket acceptor implementation.
  static impl_type null()
  {
    return invalid_socket;
  }

  // Constructor.
  reactive_socket_acceptor_service(Demuxer& d)
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

  // Open a new socket acceptor implementation.
  template <typename Protocol, typename Error_Handler>
  void open(impl_type& impl, const Protocol& protocol,
      Error_Handler error_handler)
  {
    socket_holder sock(socket_ops::socket(protocol.family(), protocol.type(),
          protocol.protocol()));
    if (sock.get() == invalid_socket)
      error_handler(asio::error(socket_ops::get_error()));
    else
      impl = sock.release();
  }

  // Bind the socket acceptor to the specified local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void bind(impl_type& impl, const Endpoint& endpoint,
      Error_Handler error_handler)
  {
    if (socket_ops::bind(impl, endpoint.native_data(),
          endpoint.native_size()) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Place the socket acceptor into the state where it will listen for new
  // connections.
  template <typename Error_Handler>
  void listen(impl_type& impl, int backlog, Error_Handler error_handler)
  {
    if (backlog == 0)
      backlog = SOMAXCONN;

    if (socket_ops::listen(impl, backlog) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Close a socket acceptor implementation.
  void close(impl_type& impl)
  {
    if (impl != null())
    {
      reactor_.close_descriptor(impl, socket_ops::close);
      impl = null();
    }
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
  void get_option(impl_type& impl, Option& option, Error_Handler error_handler)
  {
    size_t size = option.size();
    if (socket_ops::getsockopt(impl, option.level(), option.name(),
          option.data(), &size))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = endpoint.native_size();
    if (socket_ops::getsockname(impl, endpoint.native_data(), &addr_len))
      error_handler(asio::error(socket_ops::get_error()));
    endpoint.native_size(addr_len);
  }

  // Accept a new connection.
  template <typename Stream_Socket_Service, typename Error_Handler>
  void accept(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      Error_Handler error_handler)
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
  template <typename Stream_Socket_Service, typename Endpoint,
      typename Error_Handler>
  void accept(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      Endpoint& peer_endpoint, Error_Handler error_handler)
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

  template <typename Stream_Socket_Service, typename Handler>
  class accept_handler
  {
  public:
    accept_handler(impl_type impl, Demuxer& demuxer,
        basic_stream_socket<Stream_Socket_Service>& peer, Handler handler)
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
    basic_stream_socket<Stream_Socket_Service>& peer_;
    Handler handler_;
  };

  // Start an asynchronous accept. The peer_socket object must be valid until
  // the accept's handler is invoked.
  template <typename Stream_Socket_Service, typename Handler>
  void async_accept(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer, Handler handler)
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
          accept_handler<Stream_Socket_Service, Handler>(
            impl, demuxer_, peer, handler));
    }
  }

  template <typename Stream_Socket_Service, typename Endpoint,
      typename Handler>
  class accept_endp_handler
  {
  public:
    accept_endp_handler(impl_type impl, Demuxer& demuxer,
        basic_stream_socket<Stream_Socket_Service>& peer,
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
    basic_stream_socket<Stream_Socket_Service>& peer_;
    Endpoint& peer_endpoint_;
    Handler handler_;
  };

  // Start an asynchronous accept. The peer_socket and peer_endpoint objects
  // must be valid until the accept's handler is invoked.
  template <typename Stream_Socket_Service, typename Endpoint,
      typename Handler>
  void async_accept_endpoint(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
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
          accept_endp_handler<Stream_Socket_Service, Endpoint, Handler>(
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

#endif // ASIO_DETAIL_REACTIVE_SOCKET_ACCEPTOR_SERVICE_HPP
