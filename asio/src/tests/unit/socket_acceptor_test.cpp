//
// socket_acceptor_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/socket_acceptor.hpp"

#include <boost/bind.hpp>
#include <cstring>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void handle_accept(const error& err)
{
  BOOST_CHECK(!err);
}

void handle_connect(const error& err)
{
  BOOST_CHECK(!err);
}

void socket_acceptor_test()
{
  demuxer d;

  socket_acceptor acceptor(d, ipv4::tcp::endpoint(0));
  ipv4::tcp::endpoint server_endpoint;
  acceptor.get_local_endpoint(server_endpoint);
  server_endpoint.address(ipv4::address::loopback());

  stream_socket client_side_socket(d);
  stream_socket server_side_socket(d);

  client_side_socket.connect(server_endpoint);
  acceptor.accept(server_side_socket);

  client_side_socket.close();
  server_side_socket.close();

  client_side_socket.connect(server_endpoint);
  ipv4::tcp::endpoint client_endpoint;
  acceptor.accept_endpoint(server_side_socket, client_endpoint);

  ipv4::tcp::endpoint client_side_local_endpoint;
  client_side_socket.get_local_endpoint(client_side_local_endpoint);
  BOOST_CHECK(client_side_local_endpoint.port() == client_endpoint.port());

  ipv4::tcp::endpoint server_side_remote_endpoint;
  server_side_socket.get_remote_endpoint(server_side_remote_endpoint);
  BOOST_CHECK(server_side_remote_endpoint.port() == client_endpoint.port());

  client_side_socket.close();
  server_side_socket.close();

  acceptor.async_accept(server_side_socket, handle_accept);
  client_side_socket.async_connect(server_endpoint, handle_connect);

  d.run();

  client_side_socket.close();
  server_side_socket.close();

  acceptor.async_accept_endpoint(server_side_socket, client_endpoint,
      handle_accept);
  client_side_socket.async_connect(server_endpoint, handle_connect);

  d.reset();
  d.run();

  client_side_socket.get_local_endpoint(client_side_local_endpoint);
  BOOST_CHECK(client_side_local_endpoint.port() == client_endpoint.port());

  server_side_socket.get_remote_endpoint(server_side_remote_endpoint);
  BOOST_CHECK(server_side_remote_endpoint.port() == client_endpoint.port());
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("socket_acceptor");
  test->add(BOOST_TEST_CASE(&socket_acceptor_test));
  return test;
}
