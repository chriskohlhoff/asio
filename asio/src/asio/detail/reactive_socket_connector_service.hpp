//
// reactive_socket_connector_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_REACTIVE_SOCKET_CONNECTOR_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_CONNECTOR_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <set>
#include <boost/bind.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_stream_socket.hpp"
#include "asio/completion_context.hpp"
#include "asio/generic_address.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_address.hpp"
#include "asio/socket_error.hpp"
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

    // Add a socket to the set.
    void add_socket(socket_type s)
    {
      boost::mutex::scoped_lock lock(mutex_);
      sockets_.insert(s);
    }

    // Remove a socket from the set.
    void remove_socket(socket_type s)
    {
      boost::mutex::scoped_lock lock(mutex_);
      sockets_.erase(s);
    }

    // Get a copy of all sockets in the set.
    void get_sockets(socket_set& sockets) const
    {
      boost::mutex::scoped_lock lock(mutex_);
      sockets = sockets_;
    }

  private:
    // Mutex to protect access to the internal data.
    mutable boost::mutex mutex_;

    // The sockets currently contained in the set.
    socket_set sockets_;
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

  // Create a new socket connector implementation.
  void create(impl_type& impl)
  {
    impl = new connector_impl;
  }

  // Destroy a stream socket implementation.
  void destroy(impl_type& impl)
  {
    if (impl != null())
    {
      typename connector_impl::socket_set sockets;
      impl->get_sockets(sockets);
      typename connector_impl::socket_set::iterator i = sockets.begin();
      while (i != sockets.end())
        reactor_.close_descriptor(*i++);
      delete impl;
      impl = null();
    }
  }

  // Connect the given socket to the peer at the specified address. Throws a
  // socket_error exception on error.
  template <typename Stream_Socket_Service>
  void connect(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      const socket_address& peer_address)
  {
    // We cannot connect a socket that is already open.
    if (peer.impl() != invalid_socket)
      boost::throw_exception(socket_error(socket_error::already_connected));

    // Create a new socket for the connection. This will not be put into the
    // stream_socket object until the connection has beenestablished.
    socket_holder sock(socket_ops::socket(peer_address.family(), SOCK_STREAM,
          IPPROTO_TCP));
    if (sock.get() == invalid_socket)
      boost::throw_exception(socket_error(socket_ops::get_error()));

    // Perform the connect operation itself.
    impl->add_socket(sock.get());
    int result = socket_ops::connect(sock.get(), peer_address.native_address(),
        peer_address.native_size());
    impl->remove_socket(sock.get());
    if (result == socket_error_retval)
      boost::throw_exception(socket_error(socket_ops::get_error()));

    // Connection was successful. The stream_socket object will now take
    // ownership of the newly connected native socket handle.
    peer.set_impl(sock.release());
  }

  template <typename Stream_Socket_Service, typename Handler>
  class connect_handler
  {
  public:
    connect_handler(impl_type impl, socket_type new_socket, demuxer& demuxer,
        basic_stream_socket<Stream_Socket_Service>& peer,
        const socket_address& peer_address, Handler handler,
        completion_context& context)
      : impl_(impl),
        new_socket_(new_socket),
        demuxer_(demuxer),
        peer_(peer),
        peer_address_(peer_address),
        handler_(handler),
        context_(context)
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
        demuxer_.operation_completed(boost::bind(handler_, error), context_);
        return;
      }

      // If connection failed then post a completion with the error code.
      if (connect_error)
      {
        socket_error error(connect_error);
        demuxer_.operation_completed(boost::bind(handler_, error), context_);
        return;
      }

      // Make the socket blocking again (the default).
      ioctl_arg_type non_blocking = 0;
      if (socket_ops::ioctl(new_socket_, FIONBIO, &non_blocking))
      {
        socket_error error(socket_ops::get_error());
        demuxer_.operation_completed(boost::bind(handler_, error), context_);
        return;
      }

      // Post the result of the successful connection operation.
      peer_.set_impl(new_socket_);
      new_socket_holder.release();
      socket_error error(socket_error::success);
      demuxer_.operation_completed(boost::bind(handler_, error), context_);
    }

    void do_cancel()
    {
      socket_holder new_socket_holder(new_socket_);
      impl_->remove_socket(new_socket_);
      socket_error error(socket_error::operation_aborted);
      demuxer_.operation_completed(boost::bind(handler_, error), context_);
    }

  private:
    impl_type impl_;
    socket_type new_socket_;
    demuxer& demuxer_;
    basic_stream_socket<Stream_Socket_Service>& peer_;
    generic_address peer_address_;
    Handler handler_;
    completion_context& context_;
  };

  // Start an asynchronous connect. The peer socket object must be valid until
  // the connect's completion handler is invoked.
  template <typename Stream_Socket_Service, typename Handler>
  void async_connect(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      const socket_address& peer_address, Handler handler,
      completion_context& context)
  {
    if (peer.impl() != invalid_socket)
    {
      socket_error error(socket_error::already_connected);
      demuxer_.operation_immediate(boost::bind(handler, error));
      return;
    }

    // Create a new socket for the connection. This will not be put into the
    // stream_socket object until the connection has beenestablished.
    socket_holder new_socket(socket_ops::socket(peer_address.family(),
          SOCK_STREAM, IPPROTO_TCP));
    if (new_socket.get() == invalid_socket)
    {
      socket_error error(socket_ops::get_error());
      demuxer_.operation_immediate(boost::bind(handler, error), context);
      return;
    }

    // Mark the socket as non-blocking so that the connection will take place
    // asynchronously.
    ioctl_arg_type non_blocking = 1;
    if (socket_ops::ioctl(new_socket.get(), FIONBIO, &non_blocking))
    {
      socket_error error(socket_ops::get_error());
      demuxer_.operation_immediate(boost::bind(handler, error), context);
      return;
    }

    // Start the connect operation.
    if (socket_ops::connect(new_socket.get(), peer_address.native_address(),
          peer_address.native_size()) == 0)
    {
      // The connect operation has finished successfully so we need to post the
      // completion immediately.
      peer.set_impl(new_socket.release());
      socket_error error(socket_error::success);
      demuxer_.operation_immediate(boost::bind(handler, error), context);
    }
    else if (socket_ops::get_error() == socket_error::in_progress
        || socket_ops::get_error() == socket_error::would_block)
    {
      // The connection is happening in the background, and we need to wait
      // until the socket becomes writeable.
      impl->add_socket(new_socket.get());
      demuxer_.operation_started();
      reactor_.start_write_op(new_socket.get(),
          connect_handler<Stream_Socket_Service, Handler>(impl,
            new_socket.get(), demuxer_, peer, peer_address, handler, context));
      new_socket.release();
    }
    else
    {
      // The connect operation has failed, so post completion immediately.
      socket_error error(socket_ops::get_error());
      demuxer_.operation_immediate(boost::bind(handler, error), context);
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

#endif // ASIO_DETAIL_REACTIVE_SOCKET_CONNECTOR_SERVICE_HPP
