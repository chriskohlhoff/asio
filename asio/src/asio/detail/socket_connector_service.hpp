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

#include "asio/detail/push_options.hpp"
#include <boost/function.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_stream_socket.hpp"
#include "asio/service.hpp"
#include "asio/service_type_id.hpp"
#include "asio/detail/stream_socket_service.hpp"

namespace asio { class completion_context; }
namespace asio { class socket_address; }
namespace asio { class socket_error; }

namespace asio {
namespace detail {

class socket_connector_impl;

class socket_connector_service
  : public virtual service
{
public:
  // The service type id.
  static const service_type_id id;

  // The native type of the socket connector. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_connector_impl* impl_type;

  // The type of the stream sockets that will be connected.
  typedef basic_stream_socket<stream_socket_service> peer_type;

  // The value to use for uninitialised implementations.
  static const impl_type invalid_impl;

  // Create a new socket connector implementation.
  void create(impl_type& impl);

  // Destroy a socket connector implementation.
  void destroy(impl_type& impl);

  // Connect the given socket to the peer at the specified address. Throws a
  // socket_error exception on error.
  void connect(impl_type& impl, peer_type& peer_socket,
      const socket_address& peer_address);

  // The type of a handler called when the asynchronous connect completes. The
  // only argument is the error code.
  typedef boost::function1<void, const socket_error&> connect_handler;

  // Start an asynchronous connect. The peer_socket object must be valid until
  // the connect's completion handler is invoked.
  void async_connect(impl_type& impl, peer_type& peer_socket,
      const socket_address& peer_address, const connect_handler& handler,
      completion_context& context);

private:
  // Destroy a socket connector implementation.
  virtual void do_socket_connector_destroy(impl_type& impl) = 0;

  // Start an asynchronous connect.
  virtual void do_socket_connector_async_connect(impl_type& impl,
      peer_type& peer_socket, const socket_address& peer_address,
      const connect_handler& handler, completion_context& context) = 0;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_CONNECTOR_SERVICE_HPP
