//
// socket_connector.hpp
// ~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_SOCKET_CONNECTOR_HPP
#define ASIO_SOCKET_CONNECTOR_HPP

#include <boost/function.hpp>
#include <boost/noncopyable.hpp>
#include "asio/completion_context.hpp"
#include "asio/stream_socket.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

class demuxer;
class socket_connector_service;
class socket_address;
class socket_error;
namespace detail { class socket_connector_impl; }

/// The socket_connector class is used to connect a socket to a remote
/// endpoint.
class socket_connector
  : private boost::noncopyable
{
public:
  /// The native type of the socket connector. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef detail::socket_connector_impl* native_type;

  /// Constructor a connector. The connector is automatically opened.
  explicit socket_connector(demuxer& d);

  /// Destructor.
  ~socket_connector();

  /// Open the connector.
  void open();

  /// Close the connector.
  void close();

  /// Get the underlying handle in the native type.
  native_type native_handle() const;

  /// Connect the given socket to the peer at the specified address. Throws a
  /// socket_error exception on failure.
  template <typename Stream>
  void connect(Stream& peer_socket, const socket_address& peer_address)
  {
    connect_i(peer_socket.lowest_layer(), peer_address);
  }

  /// The type of a handler called when the asynchronous connect completes. The
  /// only argument is the error code.
  typedef boost::function1<void, const socket_error&> connect_handler;

  /// Start an asynchronous connect. The peer_socket object must be valid until
  /// the connect's completion handler is invoked.
  template <typename Stream>
  void async_connect(Stream& peer_socket,
      const socket_address& peer_address, const connect_handler& handler)
  {
    async_connect_i(peer_socket.lowest_layer(), peer_address, handler);
  }

  /// Start an asynchronous connect. The peer_socket object must be valid until
  /// the connect's completion handler is invoked.
  template <typename Stream>
  void async_connect(Stream& peer_socket,
      const socket_address& peer_address, const connect_handler& handler,
      completion_context& context)
  {
    async_connect_i(peer_socket.lowest_layer(), peer_address, handler,
        context);
  }

private:
  /// The socket_connector_service is permitted to call the associate()
  /// function.
  friend class socket_connector_service;

  // Connect the given socket to the peer at the specified address. Throws a
  // socket_error exception on failure.
  void connect_i(stream_socket& peer_socket,
      const socket_address& peer_address);

  // Start an asynchronous connect. The peer_socket object must be valid until
  // the connect's completion handler is invoked.
  void async_connect_i(stream_socket& peer_socket,
      const socket_address& peer_address, const connect_handler& handler,
      completion_context& context = completion_context::null());

  /// Associate the given stream socket with the native socket handle.
  void associate(stream_socket& peer_socket,
      stream_socket::native_type handle);

  /// The backend service implementation.
  socket_connector_service& service_;

  /// The underlying implementation.
  detail::socket_connector_impl* impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_CONNECTOR_HPP
