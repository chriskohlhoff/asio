//
// socket_acceptor.hpp
// ~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_SOCKET_ACCEPTOR_HPP
#define ASIO_SOCKET_ACCEPTOR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/function.hpp>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/completion_context.hpp"
#include "asio/stream_socket.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {

class demuxer;
class socket_acceptor_service;
class socket_address;
class socket_error;

/// The socket_acceptor class is used for accepting new socket connections.
class socket_acceptor
  : private boost::noncopyable
{
public:
  /// The native type of the socket acceptor. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef detail::socket_type native_type;

  /// Constructor an acceptor without opening it. The acceptor needs to be
  /// opened before it can accept new connections.
  explicit socket_acceptor(demuxer& d);

  /// Construct an acceptor opened on the given address and with the listen
  /// queue set to the given number of connections.
  socket_acceptor(demuxer& d, const socket_address& addr,
      int listen_queue = SOMAXCONN);

  /// Destructor.
  ~socket_acceptor();

  /// Open the acceptor using the given address and length of the listen queue.
  void open(const socket_address& addr, int listen_queue = SOMAXCONN);

  /// Close the acceptor.
  void close();

  /// Get the underlying handle in the native type.
  native_type native_handle() const;

  /// Accept a new connection. Throws a socket_error exception on failure.
  template <typename Stream>
  void accept(Stream& peer_socket)
  {
    accept_i(peer_socket.lowest_layer());
  }

  /// Accept a new connection. Throws a socket_error exception on failure.
  template <typename Stream>
  void accept(Stream& peer_socket, socket_address& peer_address)
  {
    accept_i(peer_socket.lowest_layer(), peer_address);
  }

  /// The type of a handler called when the asynchronous accept completes. The
  /// only argument is the error code.
  typedef boost::function1<void, const socket_error&> accept_handler;

  /// Start an asynchronous accept. The peer_socket object must be valid until
  /// the accept's completion handler is invoked.
  template <typename Stream>
  void async_accept(Stream& peer_socket, const accept_handler& handler)
  {
    async_accept_i(peer_socket.lowest_layer(), handler);
  }

  /// Start an asynchronous accept. The peer_socket object must be valid until
  /// the accept's completion handler is invoked.
  template <typename Stream>
  void async_accept(Stream& peer_socket, const accept_handler& handler,
      completion_context& context)
  {
    async_accept_i(peer_socket.lowest_layer(), handler, context);
  }

  /// Start an asynchronous accept. The peer_socket and peer_address objects
  /// must be valid until the accept's completion handler is invoked.
  template <typename Stream>
  void async_accept(Stream& peer_socket, socket_address& peer_address,
      const accept_handler& handler)
  {
    async_accept_i(peer_socket.lowest_layer(), peer_address, handler);
  }

  /// Start an asynchronous accept. The peer_socket and peer_address objects
  /// must be valid until the accept's completion handler is invoked.
  template <typename Stream>
  void async_accept(Stream& peer_socket, socket_address& peer_address,
      const accept_handler& handler, completion_context& context)
  {
    async_accept_i(peer_socket.lowest_layer(), peer_address, handler, context);
  }

private:
  /// The socket_acceptor_service class is permitted to call the associate()
  /// function.
  friend class socket_acceptor_service;

  // Accept a new connection. Throws a socket_error exception on failure.
  void accept_i(stream_socket& peer_socket);

  // Accept a new connection. Throws a socket_error exception on failure.
  void accept_i(stream_socket& peer_socket, socket_address& peer_address);

  // Start an asynchronous accept. The peer_socket object must be valid until
  // the accept's completion handler is invoked.
  void async_accept_i(stream_socket& peer_socket,
      const accept_handler& handler,
      completion_context& context = completion_context::null());

  // Start an asynchronous accept. The peer_socket and peer_address objects
  // must be valid until the accept's completion handler is invoked.
  void async_accept_i(stream_socket& peer_socket, socket_address& peer_address,
      const accept_handler& handler,
      completion_context& context = completion_context::null());

  /// Associate the given stream socket with the native socket handle.
  void associate(stream_socket& peer_socket,
      stream_socket::native_type handle);

  /// The backend service implementation.
  socket_acceptor_service& service_;

  /// The underlying native handle.
  native_type handle_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_ACCEPTOR_HPP
