//
// socket_connector_service.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/socket_connector_service.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/socket_address.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/socket_connector_impl.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {
namespace detail {

const service_type_id socket_connector_service::id;

const socket_connector_service::impl_type
socket_connector_service::invalid_impl = 0;

void
socket_connector_service::
create(
    impl_type& impl)
{
  impl = new socket_connector_impl;
}

void
socket_connector_service::
destroy(impl_type& impl)
{
  do_socket_connector_destroy(impl);
}

void
socket_connector_service::
connect(
    impl_type& impl,
    peer_type& peer_socket,
    const socket_address& peer_address)
{
  // We cannot connect a socket that is already open.
  if (peer_socket.impl() != invalid_socket)
    boost::throw_exception(socket_error(socket_error::already_connected));

  // Create a new socket for the connection. This will not be put into the
  // stream_socket object until the connection has beenestablished.
  socket_holder sock(socket_ops::socket(peer_address.family(), SOCK_STREAM,
        IPPROTO_TCP));
  if (sock.get() == invalid_socket)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  // Perform the connect operation itself.
  impl->add_socket(sock.get());
  int result = socket_ops::connect(sock.get(), peer_address.native_address(),
      peer_address.native_size());
  impl->remove_socket(sock.get());
  if (result == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  // Connection was successful. The stream_socket object will now take
  // ownership of the newly connected native socket handle.
  peer_socket.set_impl(sock.release());
}

void
socket_connector_service::
async_connect(
    impl_type& impl,
    peer_type& peer_socket,
    const socket_address& peer_address,
    const connect_handler& handler,
    completion_context& context)
{
  do_socket_connector_async_connect(impl, peer_socket, peer_address, handler,
      context);
}

} // namespace detail
} // namespace asio
