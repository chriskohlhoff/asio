//
// dgram_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DGRAM_SOCKET_SERVICE_HPP
#define ASIO_DGRAM_SOCKET_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/service.hpp"
#include "asio/service_type_id.hpp"
#include "asio/dgram_socket.hpp"

namespace asio {

/// The dgram_socket_service class is a base class for service implementations
/// that provide the functionality required by the dgram_socket class.
class dgram_socket_service
  : public virtual service
{
public:
  typedef dgram_socket::sendto_handler sendto_handler;
  typedef dgram_socket::recvfrom_handler recvfrom_handler;

  /// The service type id.
  static const service_type_id id;

  /// Register a new dgram socket with the service. This should be called only
  /// after the socket has been opened.
  virtual void register_dgram_socket(dgram_socket& socket) = 0;

  /// Remove a dgram socket registration from the service. This should be
  /// called immediately before the socket is closed.
  virtual void deregister_dgram_socket(dgram_socket& socket) = 0;

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  virtual void async_dgram_socket_sendto(dgram_socket& socket,
      const void* data, size_t length, const socket_address& destination,
      const sendto_handler& handler, completion_context& context) = 0;

  /// Start an asynchronous receive. The buffer for the data being received and
  /// the sender_address obejct must both be valid for the lifetime of the
  /// asynchronous operation.
  virtual void async_dgram_socket_recvfrom(dgram_socket& socket, void* data,
      size_t max_length, socket_address& sender_address,
      const recvfrom_handler& handler, completion_context& context) = 0;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DGRAM_SOCKET_SERVICE_HPP
