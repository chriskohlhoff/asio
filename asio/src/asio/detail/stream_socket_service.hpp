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

#ifndef ASIO_DETAIL_STREAM_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_STREAM_SOCKET_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/function.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/service.hpp"
#include "asio/service_type_id.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio { class completion_context; }
namespace asio { class socket_error; }

namespace asio {
namespace detail {

class stream_socket_service
  : public virtual service
{
public:
  // The service type id.
  static const service_type_id id;

  // The native type of the stream socket. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_type impl_type;

  // The value to use for uninitialised implementations.
  static const impl_type invalid_impl;

  // Create a new socket connector implementation.
  void create(impl_type& impl, impl_type new_impl);

  // Destroy a socket connector implementation.
  void destroy(impl_type& impl);

  // Send the given data to the peer. Returns the number of bytes sent or
  // 0 if the connection was closed cleanly. Throws a socket_error exception
  // on failure.
  size_t send(impl_type& impl, const void* data, size_t length);

  // The handler when a send operation is completed. The first argument is the
  // error code, the second is the number of bytes sent.
  typedef boost::function2<void, const socket_error&, size_t> send_handler;

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  void async_send(impl_type& impl, const void* data, size_t length,
      const send_handler& handler, completion_context& context);

  // Send all of the given data to the peer before returning. Returns the
  // number of bytes sent on the last send or 0 if the connection was closed
  // cleanly. Throws a socket_error exception on failure.
  size_t send_n(impl_type& impl, const void* data, size_t length,
      size_t* total_bytes_sent);

  // The handler when a send_n operation is completed. The first argument is
  // the error code, the second is the total number of bytes sent, and the
  // third is the number of bytes sent in the last send operation.
  typedef boost::function3<void, const socket_error&, size_t, size_t>
    send_n_handler;

  // Start an asynchronous send that will not return until all of the data has
  // been sent or an error occurs. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  void async_send_n(impl_type& impl, const void* data, size_t length,
      const send_n_handler& handler, completion_context& context);

  // Receive some data from the peer. Returns the number of bytes received or
  // 0 if the connection was closed cleanly. Throws a socket_error exception
  // on failure.
  size_t recv(impl_type& impl, void* data, size_t max_length);

  // The handler when a recv operation is completed. The first argument is the
  // error code, the second is the number of bytes received.
  typedef boost::function2<void, const socket_error&, size_t> recv_handler;

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  void async_recv(impl_type& impl, void* data, size_t max_length,
      const recv_handler& handler, completion_context& context);

  // Receive the specified amount of data from the peer. Returns the number of
  // bytes received on the last recv call or 0 if the connection
  // was closed cleanly. Throws a socket_error exception on failure.
  size_t recv_n(impl_type& impl, void* data, size_t length,
      size_t* total_bytes_recvd);

  // The handler when a recv_n operation is completed. The first argument is
  // the error code, the second is the number of bytes received, the third is
  // the number of bytes received in the last recv operation.
  typedef boost::function3<void, const socket_error&, size_t, size_t>
    recv_n_handler;

  // Start an asynchronous receive that will not return until the specified
  // number of bytes has been received or an error occurs. The buffer for the
  // data being received must be valid for the lifetime of the asynchronous
  // operation.
  void async_recv_n(impl_type& impl, void* data, size_t length,
      const recv_n_handler& handler, completion_context& context);

private:
  // Create a new socket connector implementation.
  virtual void do_stream_socket_create(impl_type& impl,
      impl_type new_impl) = 0;

  // Destroy a socket connector implementation.
  virtual void do_stream_socket_destroy(impl_type& impl) = 0;

  // Start an asynchronous send.
  virtual void do_stream_socket_async_send(impl_type& impl, const void* data,
      size_t length, const send_handler& handler,
      completion_context& context) = 0;

  // Start an asynchronous send that will not return until all of the data has
  // been sent or an error occurs.
  virtual void do_stream_socket_async_send_n(impl_type& impl, const void* data,
      size_t length, const send_n_handler& handler,
      completion_context& context) = 0;

  // Start an asynchronous receive.
  virtual void do_stream_socket_async_recv(impl_type& impl, void* data,
      size_t max_length, const recv_handler& handler,
      completion_context& context) = 0;

  // Start an asynchronous receive that will not return until the specified
  // number of bytes has been received or an error occurs.
  virtual void do_stream_socket_async_recv_n(impl_type& impl, void* data,
      size_t length, const recv_n_handler& handler,
      completion_context& context) = 0;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_STREAM_SOCKET_SERVICE_HPP
