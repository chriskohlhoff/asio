//
// stream_socket.hpp
// ~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_STREAM_SOCKET_HPP
#define ASIO_STREAM_SOCKET_HPP

#include <boost/function.hpp>
#include <boost/noncopyable.hpp>
#include "asio/completion_context.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

class demuxer;
class socket_acceptor;
class socket_connector;
class socket_error;
class stream_socket_service;

/// The stream_socket class provides asynchronous and blocking stream-oriented
/// socket functionality.
class stream_socket
  : private boost::noncopyable
{
public:
  /// The native type of the socket acceptor. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef detail::socket_type native_type;

  /// A stream_socket is always the lowest layer.
  typedef stream_socket lowest_layer_type;

  /// Construct a stream_socket without opening it. The socket needs to be
  /// connected or accepted before data can be sent or received on it.
  explicit stream_socket(demuxer& d);

  /// Destructor.
  ~stream_socket();

  /// Close the socket.
  void close();

  /// Get a reference to the lowest layer.
  lowest_layer_type& lowest_layer();

  /// Get the underlying handle in the native type.
  native_type native_handle() const;

  /// Send the given data to the peer. Returns the number of bytes sent or
  /// 0 if the connection was closed cleanly. Throws a socket_error exception
  /// on failure.
  size_t send(const void* data, size_t length);

  /// The handler when a send operation is completed. The first argument is the
  /// error code, the second is the number of bytes sent.
  typedef boost::function2<void, const socket_error&, size_t> send_handler;

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  void async_send(const void* data, size_t length,
      const send_handler& handler,
      completion_context& context = completion_context::null());

  /// Send all of the given data to the peer before returning. Returns the
  /// number of bytes sent on the last send or 0 if the connection was closed
  /// cleanly. Throws a socket_error exception on failure.
  size_t send_n(const void* data, size_t length, size_t* total_bytes_sent = 0);

  /// The handler when a send_n operation is completed. The first argument is
  /// the error code, the second is the total number of bytes sent, and the
  /// third is the number of bytes sent in the last send operation.
  typedef boost::function3<void, const socket_error&, size_t, size_t>
    send_n_handler;

  /// Start an asynchronous send that will not return until all of the data has
  /// been sent or an error occurs. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  void async_send_n(const void* data, size_t length,
      const send_n_handler& handler,
      completion_context& context = completion_context::null());

  /// Receive some data from the peer. Returns the number of bytes received or
  /// 0 if the connection was closed cleanly. Throws a socket_error exception
  /// on failure.
  size_t recv(void* data, size_t max_length);

  /// The handler when a recv operation is completed. The first argument is the
  /// error code, the second is the number of bytes received.
  typedef boost::function2<void, const socket_error&, size_t> recv_handler;

  /// Start an asynchronous receive. The buffer for the data being received must
  /// be valid for the lifetime of the asynchronous operation.
  void async_recv(void* data, size_t max_length,
      const recv_handler& handler,
      completion_context& context = completion_context::null());

  /// Receive the specified amount of data from the peer. Returns the number of
  /// bytes received on the last recv call or 0 if the connection
  /// was closed cleanly. Throws a socket_error exception on failure.
  size_t recv_n(void* data, size_t length, size_t* total_bytes_recvd = 0);

  /// The handler when a recv_n operation is completed. The first argument is
  /// the error code, the second is the number of bytes received, the third is
  /// the number of bytes received in the last recv operation.
  typedef boost::function3<void, const socket_error&, size_t, size_t>
    recv_n_handler;

  /// Start an asynchronous receive that will not return until the specified
  /// number of bytes has been received or an error occurs. The buffer for the
  /// data being received must be valid for the lifetime of the asynchronous
  /// operation.
  void async_recv_n(void* data, size_t length, const recv_n_handler& handler,
      completion_context& context = completion_context::null());

private:
  /// The socket_acceptor class is permitted to call the associate() function.
  friend class socket_acceptor;

  /// The socket_connector class is permitted to call the associate() function.
  friend class socket_connector;

  /// Associate the stream socket with the given native socket handle.
  void associate(native_type handle);

  /// The backend service implementation.
  stream_socket_service& service_;

  /// The underlying native handle.
  native_type handle_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_STREAM_SOCKET_HPP
