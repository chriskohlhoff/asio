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

const dgram_socket_service::impl_type
dgram_socket_service::invalid_impl = invalid_socket;

void
dgram_socket_service::
create(
    impl_type& impl,
    const socket_address& address)
{
  do_dgram_socket_create(impl, address);
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
