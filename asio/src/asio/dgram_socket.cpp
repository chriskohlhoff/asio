//
// dgram_socket.cpp
// ~~~~~~~~~~~~~~~~
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

#include "asio/dgram_socket.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include <cassert>
#include "asio/detail/pop_options.hpp"

#include "asio/demuxer.hpp"
#include "asio/dgram_socket_service.hpp"
#include "asio/socket_address.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {

dgram_socket::
dgram_socket(
    demuxer& d)
  : service_(dynamic_cast<dgram_socket_service&>(
        d.get_service(dgram_socket_service::id))),
    handle_(detail::invalid_socket)
{
}

dgram_socket::
dgram_socket(
    demuxer& d,
    const socket_address& addr)
  : service_(dynamic_cast<dgram_socket_service&>(
        d.get_service(dgram_socket_service::id))),
    handle_(detail::invalid_socket)
{
  open(addr);
}

dgram_socket::
~dgram_socket()
{
  close();
}

void
dgram_socket::
open(
    const socket_address& addr)
{
  assert(handle_ == detail::invalid_socket);

  detail::socket_holder sock(detail::socket_ops::socket(addr.family(),
      SOCK_DGRAM, IPPROTO_UDP));
  if (sock.get() == detail::invalid_socket)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));

  int reuse = 1;
  detail::socket_ops::setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &reuse,
      sizeof(reuse));

  if (detail::socket_ops::bind(sock.get(), addr.native_address(),
        addr.native_size()) == detail::socket_error_retval)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));

  handle_ = sock.release();

  service_.register_dgram_socket(*this);
}

void
dgram_socket::
close()
{
  if (handle_ != detail::invalid_socket)
  {
    service_.deregister_dgram_socket(*this);
    detail::socket_ops::close(handle_);
    handle_ = detail::invalid_socket;
  }
}

dgram_socket::native_type
dgram_socket::
native_handle() const
{
  return handle_;
}

size_t
dgram_socket::
sendto(
    const void* data,
    size_t length,
    const socket_address& destination)
{
  int bytes_sent = detail::socket_ops::sendto(handle_, data, length, 0,
      destination.native_address(), destination.native_size());
  if (bytes_sent < 0)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));
  return bytes_sent;
}

void
dgram_socket::
async_sendto(
    const void* data,
    size_t length,
    const socket_address& destination,
    const sendto_handler& handler,
    completion_context& context)
{
  service_.async_dgram_socket_sendto(*this, data, length, destination, handler,
      context);
}

size_t
dgram_socket::
recvfrom(
    void* data,
    size_t max_length,
    socket_address& sender_address)
{
  detail::socket_addr_len_type addr_len = sender_address.native_size();
  int bytes_recvd = detail::socket_ops::recvfrom(handle_, data, max_length, 0,
      sender_address.native_address(), &addr_len);
  if (bytes_recvd < 0)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));
  sender_address.native_size(addr_len);
  return bytes_recvd;
}

void
dgram_socket::
async_recvfrom(
    void* data,
    size_t max_length,
    socket_address& sender_address,
    const recvfrom_handler& handler,
    completion_context& context)
{
  service_.async_dgram_socket_recvfrom(*this, data, max_length, sender_address,
      handler, context);
}

} // namespace asio
