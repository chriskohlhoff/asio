//
// socket_acceptor_test.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#include <boost/bind.hpp>
#include <cstring>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void handle_accept(const socket_error& error)
{
  UNIT_TEST_CHECK(!error);
}

void handle_connect(const socket_error& error)
{
  UNIT_TEST_CHECK(!error);
}

void socket_acceptor_test()
{
  demuxer d;

  socket_acceptor acceptor(d, ipv4::address(0));
  ipv4::address server_addr;
  acceptor.get_local_address(server_addr);
  server_addr.host_addr_str("127.0.0.1");

  socket_connector connector(d);
  stream_socket client_side_socket(d);
  stream_socket server_side_socket(d);

  connector.connect(client_side_socket, server_addr);
  acceptor.accept(server_side_socket);

  client_side_socket.close();
  server_side_socket.close();

  connector.connect(client_side_socket, server_addr);
  ipv4::address client_addr;
  acceptor.accept_address(server_side_socket, client_addr);

  ipv4::address client_side_local_addr;
  client_side_socket.get_local_address(client_side_local_addr);
  UNIT_TEST_CHECK(client_side_local_addr.port() == client_addr.port());

  client_side_socket.close();
  server_side_socket.close();

  acceptor.async_accept(server_side_socket, handle_accept);
  connector.async_connect(client_side_socket, server_addr, handle_connect);

  d.run();

  client_side_socket.close();
  server_side_socket.close();

  acceptor.async_accept_address(server_side_socket, client_addr,
      handle_accept);
  connector.async_connect(client_side_socket, server_addr, handle_connect);

  d.reset();
  d.run();

  client_side_socket.get_local_address(client_side_local_addr);
  UNIT_TEST_CHECK(client_side_local_addr.port() == client_addr.port());
}

UNIT_TEST(socket_acceptor_test)
