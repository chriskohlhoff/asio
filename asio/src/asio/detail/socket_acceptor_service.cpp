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

namespace asio {
namespace detail {

const service_type_id socket_acceptor_service::id;

void
socket_acceptor_service::
associate_accepted_stream_socket(
    socket_acceptor& acceptor,
    stream_socket& peer_socket,
    stream_socket::native_type handle)
{
  acceptor.associate(peer_socket, handle);
}

} // namespace detail
} // namespace asio
