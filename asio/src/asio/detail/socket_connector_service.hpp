//
// socket_connector_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SOCKET_CONNECTOR_SERVICE_HPP
#define ASIO_DETAIL_SOCKET_CONNECTOR_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/service.hpp"
#include "asio/service_type_id.hpp"

namespace asio {
namespace detail {

class socket_connector_impl;

class socket_connector_service
  : public virtual service
{
public:
  typedef socket_connector::connect_handler connect_handler;

  // The service type id.
  static const service_type_id id;

  /// The native type of the socket connector. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef socket_connector_impl* impl_type;

  // Initialise a socket connector to a null implementation.
  void nullify(impl_type& impl);

  // Create a new socket connector implementation.
  void create(impl_type& impl);

  // Destroy a socket connector implementation.
  void destroy(impl_type& impl);

  // Connect the given socket to the peer at the specified address. Throws a
  // socket_error exception on failure.
  void connect(socket_type& peer_socket, const socket_address& peer_address);

  // Start an asynchronous connect. The peer_socket object must be valid until
  // the connect's completion handler is invoked.
  void async_connect(peer_socket& peer_socket,
      const socket_address& peer_address, const connect_handler& handler,
      completion_context& context = completion_context::null());

  // Register a new socket_connector with the service. This should be called
  // only after the socket connector has been opened.
  virtual void register_socket_connector(socket_connector& connector) = 0;

  // Remove a socket connector registration from the service. This should be
  // called immediately before the socket connector is closed.
  virtual void deregister_socket_connector(socket_connector& connector) = 0;

  // Start an asynchronous connect on the given socket. The peer_socket object
  // be valid until the connect's completion handler is invoked.
  virtual void async_socket_connect(socket_connector& connector,
      stream_socket& peer_socket, const socket_address& peer_address,
      const connect_handler& handler, completion_context& context) = 0;

protected:
  // Associate the given stream_socket with the underlying native handle that
  // was obtained by the connector.
  static void associate_connected_stream_socket(socket_connector& connector,
      stream_socket& peer_socket, stream_socket::native_type handle);

private:
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_CONNECTOR_SERVICE_HPP
