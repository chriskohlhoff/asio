//
// reactive_socket_connector_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_REACTIVE_SOCKET_CONNECTOR_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_CONNECTOR_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <set>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_stream_socket.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_holder.hpp"

namespace asio {
namespace detail {

template <typename Demuxer, typename Reactor>
class reactive_socket_connector_service
{
public:
  class connector_impl
    : private boost::noncopyable
  {
  public:
    typedef std::set<socket_type> socket_set;

    // Default constructor.
    connector_impl()
      : have_protocol_(false),
        family_(0),
        type_(0),
        protocol_(0)
    {
    }

    // Construct to use a specific protocol.
    connector_impl(int family, int type, int protocol)
      : have_protocol_(true),
        family_(family),
        type_(type),
        protocol_(protocol)
    {
    }

    // Whether a protocol was specified.
    bool have_protocol() const
    {
      return have_protocol_;
    }

    // Get the protocol family to use for new sockets.
    int family() const
    {
      return family_;
    }

    // Get the type to use for new sockets.
    int type() const
    {
      return type_;
    }

    // Get the protocol to use for new sockets.
    int protocol() const
    {
      return protocol_;
    }

    // Add a socket to the set.
    void add_socket(socket_type s)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      sockets_.insert(s);
    }

    // Remove a socket from the set.
    void remove_socket(socket_type s)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      sockets_.erase(s);
    }

    // Get a copy of all sockets in the set.
    void get_sockets(socket_set& sockets) const
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      sockets = sockets_;
    }

  private:
    // Mutex to protect access to the internal data.
    mutable asio::detail::mutex mutex_;

    // The sockets currently contained in the set.
    socket_set sockets_;

    // Whether a protocol has been specified.
    bool have_protocol_;

    // The protocol family to use for new sockets.
    int family_;

    // The type (e.g. SOCK_STREAM or SOCK_DGRAM) to use for new sockets.
    int type_;

    // The protocol to use for new sockets.
    int protocol_;
  };

  // The native type of the socket connector. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef connector_impl* impl_type;

  // Return a null socket connector implementation.
  static impl_type null()
  {
    return 0;
  }

  // Constructor.
  reactive_socket_connector_service(Demuxer& d)
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

  // Open a new socket connector implementation without specifying a protocol.
  void open(impl_type& impl)
  {
    impl = new connector_impl;
  }

  // Open a new socket connector implementation so that it will create sockets
  // using the specified protocol.
  template <typename Protocol>
  void open(impl_type& impl, const Protocol& protocol)
  {
    impl = new connector_impl(protocol.family(), protocol.type(),
        protocol.protocol());
  }

  // Close a socket connector implementation.
  void close(impl_type& impl)
  {
    if (impl != null())
    {
      typename connector_impl::socket_set sockets;
      impl->get_sockets(sockets);
      typename connector_impl::socket_set::iterator i = sockets.begin();
      while (i != sockets.end())
        reactor_.close_descriptor(*i++, socket_ops::close);
      delete impl;
      impl = null();
    }
  }

  // Connect the given socket to the peer at the specified address.
  template <typename Stream_Socket_Service, typename Address,
      typename Error_Handler>
  void connect(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      const Address& peer_address, Error_Handler error_handler)
  {
    // We cannot connect a socket that is already open.
    if (peer.impl() != invalid_socket)
    {
      error_handler(socket_error(socket_error::already_connected));
      return;
    }

    // Get the flags used to create the new socket.
    typedef typename Address::default_stream_protocol protocol;
    int family = impl->have_protocol() ? impl->family() : protocol().family();
    int type = impl->have_protocol() ? impl->type() : protocol().type();
    int proto = impl->have_protocol()
      ? impl->protocol() : protocol().protocol();

    // We can only connect stream sockets.
    if (type != SOCK_STREAM)
    {
      error_handler(socket_error(socket_error::invalid_argument));
      return;
    }

    // Create a new socket for the connection. This will not be put into the
    // stream_socket object until the connection has beenestablished.
    socket_holder sock(socket_ops::socket(family, type, proto));
    if (sock.get() == invalid_socket)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return;
    }

    // Perform the connect operation itself.
    impl->add_socket(sock.get());
    int result = socket_ops::connect(sock.get(), peer_address.native_address(),
        peer_address.native_size());
    impl->remove_socket(sock.get());
    if (result == socket_error_retval)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return;
    }

    // Connection was successful. The stream_socket object will now take
    // ownership of the newly connected native socket handle.
    peer.set_impl(sock.release());
  }

  template <typename Stream_Socket_Service, typename Handler>
  class connect_handler
  {
  public:
    connect_handler(impl_type impl, socket_type new_socket, Demuxer& demuxer,
        basic_stream_socket<Stream_Socket_Service>& peer, Handler handler)
      : impl_(impl),
        new_socket_(new_socket),
        demuxer_(demuxer),
        peer_(peer),
        handler_(handler)
    {
    }

    void do_operation()
    {
      // The connect operation can no longer be cancelled.
      socket_holder new_socket_holder(new_socket_);
      impl_->remove_socket(new_socket_);

      // Get the error code from the connect operation.
      int connect_error = 0;
      socket_len_type connect_error_len = sizeof(connect_error);
      if (socket_ops::getsockopt(new_socket_, SOL_SOCKET, SO_ERROR,
            &connect_error, &connect_error_len) == socket_error_retval)
      {
        socket_error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler_, error));
        demuxer_.work_finished();
        return;
      }

      // If connection failed then post the handler with the error code.
      if (connect_error)
      {
        socket_error error(connect_error);
        demuxer_.post(bind_handler(handler_, error));
        demuxer_.work_finished();
        return;
      }

      // Make the socket blocking again (the default).
      ioctl_arg_type non_blocking = 0;
      if (socket_ops::ioctl(new_socket_, FIONBIO, &non_blocking))
      {
        socket_error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler_, error));
        demuxer_.work_finished();
        return;
      }

      // Post the result of the successful connection operation.
      peer_.set_impl(new_socket_);
      new_socket_holder.release();
      socket_error error(socket_error::success);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      // The socket is closed when the reactor_.close_descriptor is called,
      // so no need to close it here.
      socket_error error(socket_error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    socket_type new_socket_;
    Demuxer& demuxer_;
    basic_stream_socket<Stream_Socket_Service>& peer_;
    Handler handler_;
  };

  // Start an asynchronous connect. The peer socket object must be valid until
  // the connect's handler is invoked.
  template <typename Stream_Socket_Service, typename Address, typename Handler>
  void async_connect(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      const Address& peer_address, Handler handler)
  {
    if (impl == null())
    {
      socket_error error(socket_error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    if (peer.impl() != invalid_socket)
    {
      socket_error error(socket_error::already_connected);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Get the flags used to create the new socket.
    typedef typename Address::default_stream_protocol protocol;
    int family = impl->have_protocol() ? impl->family() : protocol().family();
    int type = impl->have_protocol() ? impl->type() : protocol().type();
    int proto = impl->have_protocol()
      ? impl->protocol() : protocol().protocol();

    // We can only connect stream sockets.
    if (type != SOCK_STREAM)
    {
      socket_error error(socket_error::invalid_argument);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Create a new socket for the connection. This will not be put into the
    // stream_socket object until the connection has beenestablished.
    socket_holder new_socket(socket_ops::socket(family, type, proto));
    if (new_socket.get() == invalid_socket)
    {
      socket_error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Mark the socket as non-blocking so that the connection will take place
    // asynchronously.
    ioctl_arg_type non_blocking = 1;
    if (socket_ops::ioctl(new_socket.get(), FIONBIO, &non_blocking))
    {
      socket_error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Start the connect operation.
    if (socket_ops::connect(new_socket.get(), peer_address.native_address(),
          peer_address.native_size()) == 0)
    {
      // The connect operation has finished successfully so we need to post the
      // handler immediately.
      peer.set_impl(new_socket.release());
      socket_error error(socket_error::success);
      demuxer_.post(bind_handler(handler, error));
    }
    else if (socket_ops::get_error() == socket_error::in_progress
        || socket_ops::get_error() == socket_error::would_block)
    {
      // The connection is happening in the background, and we need to wait
      // until the socket becomes writeable.
      impl->add_socket(new_socket.get());
      demuxer_.work_started();
      reactor_.start_write_op(new_socket.get(),
          connect_handler<Stream_Socket_Service, Handler>(
            impl, new_socket.get(), demuxer_, peer, handler));
      new_socket.release();
    }
    else
    {
      // The connect operation has failed, so post the handler immediately.
      socket_error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
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

#endif // ASIO_DETAIL_REACTIVE_SOCKET_CONNECTOR_SERVICE_HPP
