//
// socket_acceptor_service.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/socket_acceptor_service.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/socket_address.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {
namespace detail {

const service_type_id socket_acceptor_service::id;

const socket_acceptor_service::impl_type socket_acceptor_service::invalid_impl;

void
socket_acceptor_service::
create(
    impl_type& impl,
    const socket_address& address)
{
  create(impl, address, SOMAXCONN);
}

void
socket_acceptor_service::
create(
    impl_type& impl,
    const socket_address& address,
    int listen_queue)
{
  socket_holder sock(socket_ops::socket(address.family(), SOCK_STREAM,
        IPPROTO_TCP));
  if (sock.get() == invalid_socket)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  int reuse = 1;
  socket_ops::setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &reuse,
      sizeof(reuse));

  if (socket_ops::bind(sock.get(), address.native_address(),
        address.native_size()) == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  if (socket_ops::listen(sock.get(), listen_queue) == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  impl = sock.release();
}

void
socket_acceptor_service::
destroy(
    impl_type& impl)
{
  do_socket_acceptor_destroy(impl);
}

void
socket_acceptor_service::
accept(
    impl_type& impl,
    peer_type& peer_socket)
{
  // We cannot accept a socket that is already open.
  if (peer_socket.impl() != invalid_socket)
    boost::throw_exception(socket_error(socket_error::already_connected));

  socket_type new_socket = socket_ops::accept(impl, 0, 0);
  if (int error = socket_ops::get_error())
    boost::throw_exception(socket_error(error));

  peer_socket.set_impl(new_socket);
}

void
socket_acceptor_service::
accept(
    impl_type& impl,
    peer_type& peer_socket,
    socket_address& peer_address)
{
  // We cannot accept a socket that is already open.
  if (peer_socket.impl() != invalid_socket)
    boost::throw_exception(socket_error(socket_error::already_connected));

  socket_addr_len_type addr_len = peer_address.native_size();
  socket_type new_socket = socket_ops::accept(impl,
      peer_address.native_address(), &addr_len);
  if (int error = socket_ops::get_error())
    boost::throw_exception(socket_error(error));
  peer_address.native_size(addr_len);

  peer_socket.set_impl(new_socket);
}

void
socket_acceptor_service::
async_accept(
    impl_type& impl,
    peer_type& peer_socket,
    const accept_handler& handler,
    completion_context& context)
{
  do_socket_acceptor_async_accept(impl, peer_socket, handler, context);
}

void
socket_acceptor_service::
async_accept(
    impl_type& impl,
    peer_type& peer_socket,
    socket_address& peer_address,
    const accept_handler& handler,
    completion_context& context)
{
  do_socket_acceptor_async_accept(impl, peer_socket, peer_address, handler,
      context);
}

} // namespace detail
} // namespace asio
