//
// socket_acceptor_test.cpp
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

  socket_acceptor acceptor(d, ipv4::tcp::endpoint(0));
  ipv4::tcp::endpoint server_endpoint;
  acceptor.get_local_endpoint(server_endpoint);
  server_endpoint.address(ipv4::address::loopback());

  socket_connector connector(d);
  stream_socket client_side_socket(d);
  stream_socket server_side_socket(d);

  connector.connect(client_side_socket, server_endpoint);
  acceptor.accept(server_side_socket);

  client_side_socket.close();
  server_side_socket.close();

  connector.connect(client_side_socket, server_endpoint);
  ipv4::tcp::endpoint client_endpoint;
  acceptor.accept_endpoint(server_side_socket, client_endpoint);

  ipv4::tcp::endpoint client_side_local_endpoint;
  client_side_socket.get_local_endpoint(client_side_local_endpoint);
  UNIT_TEST_CHECK(client_side_local_endpoint.port() == client_endpoint.port());

  client_side_socket.close();
  server_side_socket.close();

  acceptor.async_accept(server_side_socket, handle_accept);
  connector.async_connect(client_side_socket, server_endpoint, handle_connect);

  d.run();

  client_side_socket.close();
  server_side_socket.close();

  acceptor.async_accept_endpoint(server_side_socket, client_endpoint,
      handle_accept);
  connector.async_connect(client_side_socket, server_endpoint, handle_connect);

  d.reset();
  d.run();

  client_side_socket.get_local_endpoint(client_side_local_endpoint);
  UNIT_TEST_CHECK(client_side_local_endpoint.port() == client_endpoint.port());
}

UNIT_TEST(socket_acceptor_test)
