//
// stream_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_STREAM_SOCKET_SERVICE_HPP
#define ASIO_STREAM_SOCKET_SERVICE_HPP

#include "asio/service.hpp"
#include "asio/service_type_id.hpp"
#include "asio/stream_socket.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// The stream_socket_service class is a base class for service implementations
/// that provide the functionality required by the stream_socket class.
class stream_socket_service
  : public virtual service
{
public:
  typedef stream_socket::send_handler send_handler;
  typedef stream_socket::send_n_handler send_n_handler;
  typedef stream_socket::recv_handler recv_handler;
  typedef stream_socket::recv_n_handler recv_n_handler;

  /// The service type id.
  static const service_type_id id;

  /// Register a new stream socket with the service. This should be called only
  /// after the socket has been opened, i.e. after an accept or just before a
  /// connect.
  virtual void register_stream_socket(stream_socket& socket) = 0;

  /// Remove a stream socket registration from the service. This should be
  /// called immediately before the socket is closed.
  virtual void deregister_stream_socket(stream_socket& socket) = 0;

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  virtual void async_stream_socket_send(stream_socket& socket,
      const void* data, size_t length, const send_handler& handler,
      completion_context& context) = 0;

  /// Start an asynchronous send that will not return until all of the data has
  /// been sent or an error occurs. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  virtual void async_stream_socket_send_n(stream_socket& socket,
      const void* data, size_t length, const send_n_handler& handler,
      completion_context& context) = 0;

  /// Start an asynchronous receive. The buffer for the data being received
  /// must be valid for the lifetime of the asynchronous operation.
  virtual void async_stream_socket_recv(stream_socket& socket, void* data,
      size_t max_length, const recv_handler& handler,
      completion_context& context) = 0;

  /// Start an asynchronous receive that will not return until the specified
  /// number of bytes has been received or an error occurs. The buffer for the
  /// data being received must be valid for the lifetime of the asynchronous
  /// operation.
  virtual void async_stream_socket_recv_n(stream_socket& socket, void* data,
      size_t length, const recv_n_handler& handler,
      completion_context& context) = 0;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_STREAM_SOCKET_SERVICE_HPP
