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

#ifndef ASIO_SOCKET_CONNECTOR_SERVICE_HPP
#define ASIO_SOCKET_CONNECTOR_SERVICE_HPP

#include "asio/service.hpp"
#include "asio/service_type_id.hpp"
#include "asio/socket_connector.hpp"
#include "asio/stream_socket.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// The socket_connector_service class is a base class for service
/// implementations that provide the functionality required by the
/// socket_connector class.
class socket_connector_service
  : public virtual service
{
public:
  typedef socket_connector::connect_handler connect_handler;

  /// The service type id.
  static const service_type_id id;

  /// Register a new socket_connector with the service. This should be called
  /// only after the socket connector has been opened.
  virtual void register_socket_connector(socket_connector& connector) = 0;

  /// Remove a socket connector registration from the service. This should be
  /// called immediately before the socket connector is closed.
  virtual void deregister_socket_connector(socket_connector& connector) = 0;

  /// Start an asynchronous connect on the given socket. The peer_socket object
  /// be valid until the connect's completion handler is invoked.
  virtual void async_socket_connect(socket_connector& connector,
      stream_socket& peer_socket, const socket_address& peer_address,
      const connect_handler& handler, completion_context& context) = 0;

protected:
  /// Associate the given stream_socket with the underlying native handle that
  /// was obtained by the connector.
  static void associate_connected_stream_socket(socket_connector& connector,
      stream_socket& peer_socket, stream_socket::native_type handle);
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_CONNECTOR_SERVICE_HPP
