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

#ifndef ASIO_DETAIL_SOCKET_ACCEPTOR_SERVICE_HPP
#define ASIO_DETAIL_SOCKET_ACCEPTOR_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/function.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_stream_socket.hpp"
#include "asio/service.hpp"
#include "asio/service_type_id.hpp"
#include "asio/detail/stream_socket_service.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio { class completion_context; }
namespace asio { class socket_address; }
namespace asio { class socket_error; }

namespace asio {
namespace detail {

class socket_acceptor_service
  : public virtual service
{
public:
  // The service type id.
  static const service_type_id id;

  // The native type of the socket acceptor. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_type impl_type;

  // The value to use for uninitialised implementations.
  static const impl_type invalid_impl = invalid_socket;

  // The type of the stream sockets that will be connected.
  typedef basic_stream_socket<stream_socket_service> peer_type;

  // Create a new socket connector implementation.
  void create(impl_type& impl, const socket_address& address);

  // Create a new socket connector implementation.
  void create(impl_type& impl, const socket_address& address,
      int listen_queue);

  // Destroy a socket connector implementation.
  void destroy(impl_type& impl);

  // Accept a new connection. Throws a socket_error exception on failure.
  void accept(impl_type& impl, peer_type& peer_socket);

  // Accept a new connection. Throws a socket_error exception on failure.
  void accept(impl_type& impl, peer_type& peer_socket,
      socket_address& peer_address);

  // The type of a handler called when the asynchronous accept completes. The
  // only argument is the error code.
  typedef boost::function1<void, const socket_error&> accept_handler;

  // Start an asynchronous accept. The peer_socket object must be valid until
  // the accept's completion handler is invoked.
  void async_accept(impl_type& impl, peer_type& peer_socket,
      const accept_handler& handler, completion_context& context);

  // Start an asynchronous accept. The peer_socket and peer_address objects
  // must be valid until the accept's completion handler is invoked.
  void async_accept(impl_type& impl, peer_type& peer_socket,
      socket_address& peer_address, const accept_handler& handler,
      completion_context& context);

private:
  // Destroy a socket connector implementation.
  virtual void do_socket_acceptor_destroy(impl_type& impl) = 0;

  // Start an asynchronous accept on the given socket. The peer_socket object
  // must be valid until the accept's completion handler is invoked.
  virtual void do_socket_acceptor_async_accept(impl_type& impl,
      peer_type& peer_socket, const accept_handler& handler,
      completion_context& context) = 0;

  // Start an asynchronous accept on the given socket. The peer_socket and
  // peer_address objects must be valid until the accept's completion handler
  // is invoked.
  virtual void do_socket_acceptor_async_accept(impl_type& impl,
      peer_type& peer_socket, socket_address& peer_address,
      const accept_handler& handler, completion_context& context) = 0;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_ACCEPTOR_SERVICE_HPP
