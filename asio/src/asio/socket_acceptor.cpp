//
// socket_acceptor.cpp
// ~~~~~~~~~~~~~~~~~~~
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

#include "asio/socket_acceptor.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include <cassert>
#include "asio/detail/pop_options.hpp"

#include "asio/demuxer.hpp"
#include "asio/socket_acceptor_service.hpp"
#include "asio/socket_address.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {

socket_acceptor::
socket_acceptor(
    demuxer& d)
  : service_(dynamic_cast<socket_acceptor_service&>(
        d.get_service(socket_acceptor_service::id))),
    handle_(detail::invalid_socket)
{
}

socket_acceptor::
socket_acceptor(
    demuxer& d,
    const socket_address& addr,
    int listen_queue)
  : service_(dynamic_cast<socket_acceptor_service&>(
        d.get_service(socket_acceptor_service::id))),
    handle_(detail::invalid_socket)
{
  open(addr, listen_queue);
}

socket_acceptor::
~socket_acceptor()
{
  close();
}

void
socket_acceptor::
open(
    const socket_address& addr,
    int listen_queue)
{
  assert(handle_ == detail::invalid_socket);

  detail::socket_holder sock(detail::socket_ops::socket(addr.family(),
        SOCK_STREAM, IPPROTO_TCP));
  if (sock.get() == detail::invalid_socket)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));

  int reuse = 1;
  detail::socket_ops::setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &reuse,
      sizeof(reuse));

  if (detail::socket_ops::bind(sock.get(), addr.native_address(),
        addr.native_size()) == detail::socket_error_retval)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));

  if (detail::socket_ops::listen(sock.get(), listen_queue)
      == detail::socket_error_retval)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));

  handle_ = sock.release();

  service_.register_socket_acceptor(*this);
}

void
socket_acceptor::
close()
{
  if (handle_ != detail::invalid_socket)
  {
    service_.deregister_socket_acceptor(*this);
    detail::socket_ops::close(handle_);
    handle_ = detail::invalid_socket;
  }
}

socket_acceptor::native_type
socket_acceptor::
native_handle() const
{
  return handle_;
}

void
socket_acceptor::
accept_i(
    stream_socket& peer_socket)
{
  // We cannot accept a socket that is already open.
  if (peer_socket.native_handle() != detail::invalid_socket)
    boost::throw_exception(socket_error(socket_error::already_connected));

  detail::socket_type new_socket = detail::socket_ops::accept(handle_, 0, 0);
  if (int error = detail::socket_ops::get_error())
    boost::throw_exception(socket_error(error));
  peer_socket.associate(new_socket);
}

void
socket_acceptor::
accept_i(
    stream_socket& peer_socket,
    socket_address& peer_address)
{
  // We cannot accept a socket that is already open.
  if (peer_socket.native_handle() != detail::invalid_socket)
    boost::throw_exception(socket_error(socket_error::already_connected));

  detail::socket_addr_len_type addr_len = peer_address.native_size();
  detail::socket_type new_socket = detail::socket_ops::accept(handle_,
      peer_address.native_address(), &addr_len);
  if (int error = detail::socket_ops::get_error())
    boost::throw_exception(socket_error(error));
  peer_address.native_size(addr_len);
  peer_socket.associate(new_socket);
}

void
socket_acceptor::
async_accept_i(
    stream_socket& peer_socket,
    const accept_handler& handler,
    completion_context& context)
{
  service_.async_socket_accept(*this, peer_socket, handler, context);
}

void
socket_acceptor::
async_accept_i(
    stream_socket& peer_socket,
    socket_address& peer_address,
    const accept_handler& handler,
    completion_context& context)
{
  service_.async_socket_accept(*this, peer_socket, peer_address, handler,
      context);
}

void
socket_acceptor::
associate(
    stream_socket& peer_socket,
    stream_socket::native_type handle)
{
  peer_socket.associate(handle);
}

} // namespace asio
