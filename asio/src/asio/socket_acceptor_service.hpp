//
// socket_acceptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_SOCKET_ACCEPTOR_SERVICE_HPP
#define ASIO_SOCKET_ACCEPTOR_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/service.hpp"
#include "asio/service_type_id.hpp"
#include "asio/socket_acceptor.hpp"
#include "asio/stream_socket.hpp"

namespace asio {

/// The socket_acceptor_service class is a base class for service
/// implementations that provide the functionality required by the
/// socket_acceptor class.
class socket_acceptor_service
  : public virtual service
{
public:
  typedef socket_acceptor::accept_handler accept_handler;

  /// The service type id.
  static const service_type_id id;

  /// Register a new socket_acceptor with the service. This should be called
  /// only after the socket acceptor has been opened.
  virtual void register_socket_acceptor(socket_acceptor& acceptor) = 0;

  /// Remove a socket acceptor registration from the service. This should be
  /// called immediately before the socket acceptor is closed.
  virtual void deregister_socket_acceptor(socket_acceptor& acceptor) = 0;

  /// Start an asynchronous accept on the given socket. The peer_socket object
  /// must be valid until the accept's completion handler is invoked.
  virtual void async_socket_accept(socket_acceptor& acceptor,
      stream_socket& peer_socket, const accept_handler& handler,
      completion_context& context) = 0;

  /// Start an asynchronous accept on the given socket. The peer_socket and
  /// peer_address objects must be valid until the accept's completion handler
  /// is invoked.
  virtual void async_socket_accept(socket_acceptor& acceptor,
      stream_socket& peer_socket, socket_address& peer_address,
      const accept_handler& handler, completion_context& context) = 0;

protected:
  /// Associate the given stream_socket with the underlying native handle that
  /// was obtained by the acceptor.
  static void associate_accepted_stream_socket(socket_acceptor& acceptor,
      stream_socket& peer_socket, stream_socket::native_type handle);
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_ACCEPTOR_SERVICE_HPP
