//
// stream_socket_service.cpp
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

#include "asio/detail/stream_socket_service.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/socket_error.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {
namespace detail {

const service_type_id stream_socket_service::id;

const stream_socket_service::impl_type stream_socket_service::invalid_impl;

void
stream_socket_service::
create(
    impl_type& impl,
    impl_type new_impl)
{
  do_stream_socket_create(impl, new_impl);
}

void
stream_socket_service::
destroy(
    impl_type& impl)
{
  do_stream_socket_destroy(impl);
}

size_t
stream_socket_service::
send(
    impl_type& impl,
    const void* data,
    size_t length)
{
  int bytes_sent = socket_ops::send(impl, data, length, 0);
  if (bytes_sent < 0)
    boost::throw_exception(socket_error(socket_ops::get_error()));
  return bytes_sent;
}

void
stream_socket_service::
async_send(
    impl_type& impl,
    const void* data,
    size_t length,
    const send_handler& handler,
    completion_context& context)
{
  do_stream_socket_async_send(impl, data, length, handler, context);
}

size_t
stream_socket_service::
send_n(
    impl_type& impl,
    const void* data,
    size_t length,
    size_t* total_bytes_sent)
{
  // TODO handle non-blocking sockets using select to wait until ready.

  int bytes_sent = 0;
  size_t total_sent = 0;
  while (total_sent < length)
  {
    bytes_sent = socket_ops::send(impl,
        static_cast<const char*>(data) + total_sent, length - total_sent, 0);
    if (bytes_sent < 0)
    {
      boost::throw_exception(socket_error(socket_ops::get_error()));
    }
    else if (bytes_sent == 0)
    {
      if (total_bytes_sent)
        *total_bytes_sent = total_sent;
      return bytes_sent;
    }
    total_sent += bytes_sent;
  }
  if (total_bytes_sent)
    *total_bytes_sent = total_sent;
  return bytes_sent;
}

void
stream_socket_service::
async_send_n(
    impl_type& impl,
    const void* data,
    size_t length,
    const send_n_handler& handler,
    completion_context& context)
{
  do_stream_socket_async_send_n(impl, data, length, handler, context);
}

size_t
stream_socket_service::
recv(
    impl_type& impl,
    void* data,
    size_t max_length)
{
  int bytes_recvd = socket_ops::recv(impl, data, max_length, 0);
  if (bytes_recvd < 0)
    boost::throw_exception(socket_error(socket_ops::get_error()));
  return bytes_recvd;
}

void
stream_socket_service::
async_recv(
    impl_type& impl,
    void* data,
    size_t max_length,
    const recv_handler& handler,
    completion_context& context)
{
  do_stream_socket_async_recv(impl, data, max_length, handler, context);
}

size_t
stream_socket_service::
recv_n(
    impl_type& impl,
    void* data,
    size_t length,
    size_t* total_bytes_recvd)
{
  // TODO handle non-blocking sockets using select to wait until ready.

  int bytes_recvd = 0;
  size_t total_recvd = 0;
  while (total_recvd < length)
  {
    bytes_recvd = socket_ops::recv(impl,
        static_cast<char*>(data) + total_recvd, length - total_recvd, 0);
    if (bytes_recvd < 0)
    {
      boost::throw_exception(socket_error(socket_ops::get_error()));
    }
    else if (bytes_recvd == 0)
    {
      if (total_bytes_recvd)
        *total_bytes_recvd = total_recvd;
      return bytes_recvd;
    }
    total_recvd += bytes_recvd;
  }
  if (total_bytes_recvd)
    *total_bytes_recvd = total_recvd;
  return bytes_recvd;
}

void
stream_socket_service::
async_recv_n(
    impl_type& impl,
    void* data,
    size_t length,
    const recv_n_handler& handler,
    completion_context& context)
{
  do_stream_socket_async_recv_n(impl, data, length, handler, context);
}

} // namespace detail
} // namespace asio
