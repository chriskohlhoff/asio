//
// socket_connector.cpp
// ~~~~~~~~~~~~~~~~~~~~
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

#include "asio/socket_connector.hpp"
#include <boost/throw_exception.hpp>
#include <cassert>
#include "asio/demuxer.hpp"
#include "asio/socket_connector_service.hpp"
#include "asio/socket_address.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/socket_connector_impl.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {

socket_connector::
socket_connector(
    demuxer& d)
  : service_(dynamic_cast<socket_connector_service&>(
        d.get_service(socket_connector_service::id))),
    impl_(new detail::socket_connector_impl)
{
  service_.register_socket_connector(*this);
}

socket_connector::
~socket_connector()
{
  delete impl_;
}

void
socket_connector::
open()
{
  assert(impl_ == 0);
  impl_ = new detail::socket_connector_impl;
  service_.register_socket_connector(*this);
}

void
socket_connector::
close()
{
  if (impl_)
  {
    service_.deregister_socket_connector(*this);
    delete impl_;
    impl_ = 0;
  }
}

socket_connector::native_type
socket_connector::
native_handle() const
{
  return impl_;
}

void
socket_connector::
connect_i(
    stream_socket& peer_socket,
    const socket_address& peer_address)
{
  // We cannot connect a socket that is already open.
  if (peer_socket.native_handle() != detail::invalid_socket)
    boost::throw_exception(socket_error(socket_error::already_connected));

  // Create a new socket for the connection. This will not be put into the
  // stream_socket object until the connection has beenestablished.
  detail::socket_holder sock(detail::socket_ops::socket(peer_address.family(),
        SOCK_STREAM, IPPROTO_TCP));
  if (sock.get() == detail::invalid_socket)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));

  // Perform the connect operation itself.
  impl_->add_socket(sock.get());
  int result = detail::socket_ops::connect(sock.get(),
      peer_address.native_address(), peer_address.native_size());
  impl_->remove_socket(sock.get());
  if (result == detail::socket_error_retval)
    boost::throw_exception(socket_error(detail::socket_ops::get_error()));

  // Connection was successful. The stream_socket object will now take
  // ownership of the newly connected native socket handle.
  peer_socket.associate(sock.release());
}

void
socket_connector::
async_connect_i(
    stream_socket& peer_socket,
    const socket_address& peer_address,
    const connect_handler& handler,
    completion_context& context)
{
  service_.async_socket_connect(*this, peer_socket, peer_address, handler,
      context);
}

void
socket_connector::
associate(
    stream_socket& peer_socket,
    stream_socket::native_type handle)
{
  peer_socket.associate(handle);
}

} // namespace asio
