//
// reactive_socket_acceptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_REACTIVE_SOCKET_ACCEPTOR_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_ACCEPTOR_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_stream_socket.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_error.hpp"
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

  // Create a new stream socket implementation.
  template <typename Address, typename Error_Handler>
  void create(impl_type& impl, const Address& address, int listen_queue,
      Error_Handler error_handler)
  {
    if (listen_queue == 0)
      listen_queue = SOMAXCONN;

    socket_holder sock(socket_ops::socket(address.family(), SOCK_STREAM,
          IPPROTO_TCP));
    if (sock.get() == invalid_socket)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return;
    }

    int reuse = 1;
    socket_ops::setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &reuse,
        sizeof(reuse));

    if (socket_ops::bind(sock.get(), address.native_address(),
          address.native_size()) == socket_error_retval)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return;
    }

    if (socket_ops::listen(sock.get(), listen_queue) == socket_error_retval)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return;
    }

    impl = sock.release();
  }

  // Destroy a stream socket implementation.
  void destroy(impl_type& impl)
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
      error_handler(socket_error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(impl_type& impl, Option& option, Error_Handler error_handler)
  {
    socket_len_type size = option.size();
    if (socket_ops::getsockopt(impl, option.level(), option.name(),
          option.data(), &size))
      error_handler(socket_error(socket_ops::get_error()));
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
      error_handler(socket_error(socket_error::already_connected));
      return;
    }

    socket_type new_socket = socket_ops::accept(impl, 0, 0);
    if (int error = socket_ops::get_error())
    {
      error_handler(socket_error(error));
      return;
    }

    peer.set_impl(new_socket);
  }

  // Accept a new connection.
  template <typename Stream_Socket_Service, typename Address,
      typename Error_Handler>
  void accept(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer, Address& peer_address,
      Error_Handler error_handler)
  {
    // We cannot accept a socket that is already open.
    if (peer.impl() != invalid_socket)
    {
      error_handler(socket_error(socket_error::already_connected));
      return;
    }

    socket_addr_len_type addr_len = peer_address.native_size();
    socket_type new_socket = socket_ops::accept(impl,
        peer_address.native_address(), &addr_len);
    if (int error = socket_ops::get_error())
    {
      error_handler(socket_error(error));
      return;
    }

    peer_address.native_size(addr_len);

    peer.set_impl(new_socket);
  }

  template <typename Stream_Socket_Service, typename Handler,
      typename Completion_Context>
  class accept_handler
  {
  public:
    accept_handler(impl_type impl, Demuxer& demuxer,
        basic_stream_socket<Stream_Socket_Service>& peer, Handler handler,
        Completion_Context context)
      : impl_(impl),
        demuxer_(demuxer),
        peer_(peer),
        handler_(handler),
        context_(context)
    {
    }

    void do_operation()
    {
      socket_type new_socket = socket_ops::accept(impl_, 0, 0);
      socket_error error(new_socket == invalid_socket
          ? socket_ops::get_error() : socket_error::success);
      peer_.set_impl(new_socket);
      demuxer_.operation_completed(bind_handler(handler_, error), context_);
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.operation_completed(bind_handler(handler_, error), context_);
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    basic_stream_socket<Stream_Socket_Service>& peer_;
    Handler handler_;
    Completion_Context context_;
  };

  // Start an asynchronous accept. The peer_socket object must be valid until
  // the accept's completion handler is invoked.
  template <typename Stream_Socket_Service, typename Handler,
      typename Completion_Context>
  void async_accept(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer, Handler handler,
      Completion_Context context)
  {
    if (peer.impl() != invalid_socket)
    {
      socket_error error(socket_error::already_connected);
      demuxer_.operation_immediate(bind_handler(handler, error));
    }
    else
    {
      demuxer_.operation_started();
      reactor_.start_read_op(impl,
          accept_handler<Stream_Socket_Service, Handler, Completion_Context>(
            impl, demuxer_, peer, handler, context));
    }
  }

  template <typename Stream_Socket_Service, typename Address, typename Handler,
      typename Completion_Context>
  class accept_addr_handler
  {
  public:
    accept_addr_handler(impl_type impl, Demuxer& demuxer,
        basic_stream_socket<Stream_Socket_Service>& peer,
        Address& peer_address, Handler handler, Completion_Context context)
      : impl_(impl),
        demuxer_(demuxer),
        peer_(peer),
        peer_address_(peer_address),
        handler_(handler),
        context_(context)
    {
    }

    void do_operation()
    {
      socket_addr_len_type addr_len = peer_address_.native_size();
      socket_type new_socket = socket_ops::accept(impl_,
          peer_address_.native_address(), &addr_len);
      socket_error error(new_socket == invalid_socket
          ? socket_ops::get_error() : socket_error::success);
      peer_address_.native_size(addr_len);
      peer_.set_impl(new_socket);
      demuxer_.operation_completed(bind_handler(handler_, error), context_);
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.operation_completed(bind_handler(handler_, error), context_);
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    basic_stream_socket<Stream_Socket_Service>& peer_;
    Address& peer_address_;
    Handler handler_;
    Completion_Context context_;
  };

  // Start an asynchronous accept. The peer_socket and peer_address objects
  // must be valid until the accept's completion handler is invoked.
  template <typename Stream_Socket_Service, typename Address, typename Handler,
      typename Completion_Context>
  void async_accept_address(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      Address& peer_address, Handler handler, Completion_Context context)
  {
    if (peer.impl() != invalid_socket)
    {
      socket_error error(socket_error::already_connected);
      demuxer_.operation_immediate(bind_handler(handler, error));
    }
    else
    {
      demuxer_.operation_started();
      reactor_.start_read_op(impl,
          accept_addr_handler<Stream_Socket_Service, Address, Handler,
              Completion_Context>(impl, demuxer_, peer, peer_address, handler,
                context));
    }
  }

private:
  // The demuxer used for delivering completion notifications.
  Demuxer& demuxer_;

  // The selector that performs event demultiplexing for the provider.
  Reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_SOCKET_ACCEPTOR_SERVICE_HPP
