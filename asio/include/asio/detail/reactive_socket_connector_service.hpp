//
// reactive_socket_connector_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_SOCKET_CONNECTOR_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_CONNECTOR_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <list>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_stream_socket.hpp"
#include "asio/error.hpp"
#include "asio/service_factory.hpp"
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
  struct connection_info
  {
    socket_type sock;
    int ref_count;
  };

  typedef std::list<connection_info> connection_info_list;
  typedef typename connection_info_list::iterator connection_info_handle;

  class connector_impl
    : private boost::noncopyable
  {
  public:
    // Default constructor.
    connector_impl()
      : ref_count_(1),
        have_protocol_(false),
        family_(0),
        type_(0),
        protocol_(0)
    {
    }

    // Construct to use a specific protocol.
    connector_impl(int family, int type, int protocol)
      : ref_count_(1),
        have_protocol_(true),
        family_(family),
        type_(type),
        protocol_(protocol)
    {
    }

    // Increment reference count.
    void add_ref()
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      ++ref_count_;
    }

    // Decrement reference count and delete object if required.
    void remove_ref()
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      if (--ref_count_ == 0)
      {
        lock.unlock();
        delete this;
      }
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

    // Create the connection information associated with a connection
    // operation. Sets a reference count so that the connection_info object and
    // the connection implementation itself are not deleted until all
    // references are removed.
    connection_info_handle create_connection_info(socket_type s)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      connection_info new_info;
      new_info.sock = s;
      new_info.ref_count = 1;
      connection_info_list_.push_front(new_info);
      ++ref_count_;
      return connection_info_list_.begin();
    }

    // Add a reference to a connection info object.
    void add_connection_info_ref(connection_info_handle info)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      ++info->ref_count;
      ++ref_count_;
    }

    // Remove a reference to a connection info object. The object will be
    // removed from the list if the count has reached zero. If the connector
    // implementation reference count reaches zero then that object is deleted.
    // Returns the new reference count of the connection info object.
    int remove_connection_info_ref(connection_info_handle info)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      int connection_info_ref_count = --info->ref_count;
      if (connection_info_ref_count == 0)
        connection_info_list_.erase(info);
      if (--ref_count_ == 0)
      {
        lock.unlock();
        delete this;
      }
      return connection_info_ref_count;
    }

    // Get a copy of the connection info list.
    void get_connection_info_list(connection_info_list& info_list) const
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      info_list = connection_info_list_;
    }

  private:
    // Mutex to protect access to the internal data.
    mutable asio::detail::mutex mutex_;

    // Reference count so that the object does not go away while there are
    // outstanding connection attempts associated with it.
    int ref_count_;

    // The connection_info objects currently associated with the connector.
    connection_info_list connection_info_list_;

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
      connection_info_list info_list;
      impl->get_connection_info_list(info_list);
      connection_info_handle i = info_list.begin();
      while (i != info_list.end())
      {
        reactor_.close_descriptor(i->sock, socket_ops::close);
        ++i;
      }
      impl->remove_ref();
      impl = null();
    }
  }

  // Connect the given socket to the peer at the specified endpoint.
  template <typename Stream_Socket_Service, typename Endpoint,
      typename Error_Handler>
  void connect(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      const Endpoint& peer_endpoint, Error_Handler error_handler)
  {
    // We cannot connect a socket that is already open.
    if (peer.impl() != invalid_socket)
    {
      error_handler(asio::error(asio::error::already_connected));
      return;
    }

    // Get the flags used to create the new socket.
    int family = impl->have_protocol()
      ? impl->family() : peer_endpoint.protocol().family();
    int type = impl->have_protocol()
      ? impl->type() : peer_endpoint.protocol().type();
    int proto = impl->have_protocol()
      ? impl->protocol() : peer_endpoint.protocol().protocol();

    // We can only connect stream sockets.
    if (type != SOCK_STREAM)
    {
      error_handler(asio::error(asio::error::invalid_argument));
      return;
    }

    // Create a new socket for the connection. This will not be put into the
    // stream_socket object until the connection has beenestablished.
    socket_holder sock(socket_ops::socket(family, type, proto));
    if (sock.get() == invalid_socket)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    // Perform the connect operation itself.
    int result = socket_ops::connect(sock.get(), peer_endpoint.native_data(),
        peer_endpoint.native_size());
    if (result == socket_error_retval)
    {
      error_handler(asio::error(socket_ops::get_error()));
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
    connect_handler(impl_type impl, connection_info_handle conn_info,
        Demuxer& demuxer, Reactor& reactor,
        basic_stream_socket<Stream_Socket_Service>& peer, Handler handler)
      : impl_(impl),
        conn_info_(conn_info),
        demuxer_(demuxer),
        reactor_(reactor),
        peer_(peer),
        handler_(handler)
    {
    }

    void do_operation()
    {
      // Check whether a handler has already been called for the connection.
      // If it has, then we don't want to do anything in this handler.
      socket_type new_socket = conn_info_->sock;
      if (impl_->remove_connection_info_ref(conn_info_) == 0)
      {
        demuxer_.work_finished();
        return;
      }

      // Cancel the other reactor operation for the connection.
      reactor_.enqueue_cancel_ops_unlocked(new_socket);

      // Take ownership of the socket.
      socket_holder new_socket_holder(new_socket);

      // Get the error code from the connect operation.
      int connect_error = 0;
      size_t connect_error_len = sizeof(connect_error);
      if (socket_ops::getsockopt(new_socket, SOL_SOCKET, SO_ERROR,
            &connect_error, &connect_error_len) == socket_error_retval)
      {
        asio::error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // If connection failed then post the handler with the error code.
      if (connect_error)
      {
        asio::error error(connect_error);
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // Make the socket blocking again (the default).
      ioctl_arg_type non_blocking = 0;
      if (socket_ops::ioctl(new_socket, FIONBIO, &non_blocking))
      {
        asio::error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // Post the result of the successful connection operation.
      peer_.set_impl(new_socket);
      new_socket_holder.release();
      asio::error error(asio::error::success);
      demuxer_.post(bind_handler(handler_, error));
    }

    void do_cancel()
    {
      // Check whether a handler has already been called for the connection.
      // If it has, then we don't want to do anything in this handler.
      socket_type new_socket = conn_info_->sock;
      if (impl_->remove_connection_info_ref(conn_info_) == 0)
      {
        demuxer_.work_finished();
        return;
      }

      // Cancel the other reactor operation for the connection.
      reactor_.enqueue_cancel_ops_unlocked(new_socket);

      // The socket is closed when the reactor_.close_descriptor is called,
      // so no need to close it here.
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error));
    }

  private:
    impl_type impl_;
    connection_info_handle conn_info_;
    Demuxer& demuxer_;
    Reactor& reactor_;
    basic_stream_socket<Stream_Socket_Service>& peer_;
    Handler handler_;
  };

  // Start an asynchronous connect. The peer socket object must be valid until
  // the connect's handler is invoked.
  template <typename Stream_Socket_Service, typename Endpoint,
      typename Handler>
  void async_connect(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      const Endpoint& peer_endpoint, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    if (peer.impl() != invalid_socket)
    {
      asio::error error(asio::error::already_connected);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Get the flags used to create the new socket.
    int family = impl->have_protocol()
      ? impl->family() : peer_endpoint.protocol().family();
    int type = impl->have_protocol()
      ? impl->type() : peer_endpoint.protocol().type();
    int proto = impl->have_protocol()
      ? impl->protocol() : peer_endpoint.protocol().protocol();

    // We can only connect stream sockets.
    if (type != SOCK_STREAM)
    {
      asio::error error(asio::error::invalid_argument);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Create a new socket for the connection. This will not be put into the
    // stream_socket object until the connection has beenestablished.
    socket_holder new_socket(socket_ops::socket(family, type, proto));
    if (new_socket.get() == invalid_socket)
    {
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Mark the socket as non-blocking so that the connection will take place
    // asynchronously.
    ioctl_arg_type non_blocking = 1;
    if (socket_ops::ioctl(new_socket.get(), FIONBIO, &non_blocking))
    {
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Start the connect operation.
    if (socket_ops::connect(new_socket.get(), peer_endpoint.native_data(),
          peer_endpoint.native_size()) == 0)
    {
      // The connect operation has finished successfully so we need to post the
      // handler immediately.
      peer.set_impl(new_socket.release());
      asio::error error(asio::error::success);
      demuxer_.post(bind_handler(handler, error));
    }
    else if (socket_ops::get_error() == asio::error::in_progress
        || socket_ops::get_error() == asio::error::would_block)
    {
      // The connection is happening in the background, and we need to wait
      // until the socket becomes writeable.
      connection_info_handle conn_info
        = impl->create_connection_info(new_socket.get());
      impl->add_connection_info_ref(conn_info);
      demuxer_.work_started();
      reactor_.start_write_and_except_ops(new_socket.get(),
          connect_handler<Stream_Socket_Service, Handler>(
            impl, conn_info, demuxer_, reactor_, peer, handler));
      new_socket.release();
    }
    else
    {
      // The connect operation has failed, so post the handler immediately.
      asio::error error(socket_ops::get_error());
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
