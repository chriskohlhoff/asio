//
// stream_socket.cpp
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

#include "asio/stream_socket.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include <cassert>
#include "asio/detail/pop_options.hpp"

#include "asio/demuxer.hpp"
#include "asio/socket_error.hpp"
#include "asio/stream_socket_service.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {

stream_socket::
stream_socket(
    demuxer& d)
  : service_(dynamic_cast<stream_socket_service&>(
        d.get_service(stream_socket_service::id))),
    handle_(detail::invalid_socket)
{
}

stream_socket::
~stream_socket()
{
  close();
}

void
stream_socket::
close()
{
  if (handle_ != detail::invalid_socket)
  {
    service_.deregister_stream_socket(*this);
    detail::socket_ops::close(handle_);
    handle_ = detail::invalid_socket;
  }
}

stream_socket::lowest_layer_type&
stream_socket::
lowest_layer()
{
  return *this;
}

stream_socket::native_type
stream_socket::
native_handle() const
{
  return handle_;
}

size_t
stream_socket::
send(
    const void* data,
    size_t length)
{
  int bytes_sent = detail::socket_ops::send(handle_, data, length, 0);
  if (bytes_sent < 0)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));
  return bytes_sent;
}

void
stream_socket::
async_send(
    const void* data,
    size_t length,
    const send_handler& handler,
    completion_context& context)
{
  service_.async_stream_socket_send(*this, data, length, handler, context);
}

size_t
stream_socket::
send_n(
    const void* data,
    size_t length,
    size_t* total_bytes_sent)
{
  // TODO handle non-blocking sockets using select to wait until ready.

  int bytes_sent = 0;
  size_t total_sent = 0;
  while (total_sent < length)
  {
    bytes_sent = detail::socket_ops::send(handle_,
        static_cast<const char*>(data) + total_sent, length - total_sent, 0);
    if (bytes_sent < 0)
    {
      boost::throw_exception(socket_error(detail::socket_ops::get_error()));
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
stream_socket::
async_send_n(
    const void* data,
    size_t length,
    const send_n_handler& handler,
    completion_context& context)
{
  service_.async_stream_socket_send_n(*this, data, length, handler, context);
}

size_t
stream_socket::
recv(
    void* data,
    size_t max_length)
{
  int bytes_recvd = detail::socket_ops::recv(handle_, data, max_length, 0);
  if (bytes_recvd < 0)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));
  return bytes_recvd;
}

void
stream_socket::
async_recv(
    void* data,
    size_t max_length,
    const recv_handler& handler,
    completion_context& context)
{
  service_.async_stream_socket_recv(*this, data, max_length, handler, context);
}

size_t
stream_socket::
recv_n(
    void* data,
    size_t length,
    size_t* total_bytes_recvd)
{
  // TODO handle non-blocking sockets using select to wait until ready.

  int bytes_recvd = 0;
  size_t total_recvd = 0;
  while (total_recvd < length)
  {
    bytes_recvd = detail::socket_ops::recv(handle_,
        static_cast<char*>(data) + total_recvd, length - total_recvd, 0);
    if (bytes_recvd < 0)
    {
      boost::throw_exception(socket_error(detail::socket_ops::get_error()));
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
stream_socket::
async_recv_n(
    void* data,
    size_t length,
    const recv_n_handler& handler,
    completion_context& context)
{
  service_.async_stream_socket_recv_n(*this, data, length, handler, context);
}

void
stream_socket::
associate(
    native_type handle)
{
  assert(handle_ == detail::invalid_socket);
  handle_ = handle;
  if (handle_ != detail::invalid_socket)
    service_.register_stream_socket(*this);
}

} // namespace asio
