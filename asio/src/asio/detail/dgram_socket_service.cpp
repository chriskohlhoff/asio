//
// dgram_socket_service.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/dgram_socket_service.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/socket_address.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {
namespace detail {

const service_type_id dgram_socket_service::id;

void
dgram_socket_service::
nullify(
    impl_type& impl)
{
  impl = invalid_socket;
}

void
dgram_socket_service::
create(
    impl_type& impl,
    const socket_address& address)
{
  socket_holder sock(socket_ops::socket(address.family(), SOCK_DGRAM,
        IPPROTO_UDP));
  if (sock.get() == invalid_socket)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  int reuse = 1;
  socket_ops::setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &reuse,
      sizeof(reuse));

  if (socket_ops::bind(sock.get(), address.native_address(),
        address.native_size()) == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  impl = sock.release();
}

void
attach(
    impl_type& impl,
    impl_type new_impl)
{
  impl = new_impl;
}

void
dgram_socket_service::
destroy(
    impl_type& impl)
{
  do_dgram_socket_destroy(impl);
}

size_t
dgram_socket_service::
sendto(
    impl_type& impl,
    const void* data,
    size_t length,
    const socket_address& destination)
{
  int bytes_sent = socket_ops::sendto(impl, data, length, 0,
      destination.native_address(), destination.native_size());
  if (bytes_sent < 0)
    boost::throw_exception(socket_error(socket_ops::get_error()));
  return bytes_sent;
}

void
dgram_socket_service::
async_sendto(
    impl_type& impl,
    const void* data,
    size_t length,
    const socket_address& destination,
    const sendto_handler& handler,
    completion_context& context)
{
  do_dgram_socket_async_sendto(impl, data, length, destination, handler,
      context);
}

size_t
dgram_socket_service::
recvfrom(
    impl_type& impl,
    void* data,
    size_t max_length,
    socket_address& sender_address)
{
  socket_addr_len_type addr_len = sender_address.native_size();
  int bytes_recvd = socket_ops::recvfrom(impl, data, max_length, 0,
      sender_address.native_address(), &addr_len);
  if (bytes_recvd < 0)
    boost::throw_exception(socket_error(socket_ops::get_error()));
  sender_address.native_size(addr_len);
  return bytes_recvd;
}

void
dgram_socket_service::
async_recvfrom(
    impl_type& impl,
    void* data,
    size_t max_length,
    socket_address& sender_address,
    const recvfrom_handler& handler,
    completion_context& context)
{
  do_dgram_socket_async_recvfrom(impl, data, max_length, sender_address,
      handler, context);
}

} // namespace detail
} // namespace asio
