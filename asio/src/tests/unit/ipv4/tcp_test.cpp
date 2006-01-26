//
// tcp_test.cpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/ipv4/tcp.hpp"

#include <boost/bind.hpp>
#include <cstring>
#include "asio.hpp"
#include "../unit_test.hpp"

using namespace asio;

void handle_accept(const error& err)
{
  BOOST_CHECK(!err);
}

void handle_connect(const error& err)
{
  BOOST_CHECK(!err);
}

void ipv4_tcp_acceptor_test()
{
  io_service ios;

  ipv4::tcp::acceptor acceptor(ios, ipv4::tcp::endpoint(0));
  ipv4::tcp::endpoint server_endpoint = acceptor.local_endpoint();
  server_endpoint.address(ipv4::address::loopback());

  ipv4::tcp::socket client_side_socket(ios);
  ipv4::tcp::socket server_side_socket(ios);

  client_side_socket.connect(server_endpoint);
  acceptor.accept(server_side_socket);

  client_side_socket.close();
  server_side_socket.close();

  client_side_socket.connect(server_endpoint);
  ipv4::tcp::endpoint client_endpoint;
  acceptor.accept_endpoint(server_side_socket, client_endpoint);

  ipv4::tcp::endpoint client_side_local_endpoint
    = client_side_socket.local_endpoint();
  BOOST_CHECK(client_side_local_endpoint.port() == client_endpoint.port());

  ipv4::tcp::endpoint server_side_remote_endpoint
    = server_side_socket.remote_endpoint();
  BOOST_CHECK(server_side_remote_endpoint.port() == client_endpoint.port());

  client_side_socket.close();
  server_side_socket.close();

  acceptor.async_accept(server_side_socket, handle_accept);
  client_side_socket.async_connect(server_endpoint, handle_connect);

  ios.run();

  client_side_socket.close();
  server_side_socket.close();

  acceptor.async_accept_endpoint(server_side_socket, client_endpoint,
      handle_accept);
  client_side_socket.async_connect(server_endpoint, handle_connect);

  ios.reset();
  ios.run();

  client_side_local_endpoint = client_side_socket.local_endpoint();
  BOOST_CHECK(client_side_local_endpoint.port() == client_endpoint.port());

  server_side_remote_endpoint = server_side_socket.remote_endpoint();
  BOOST_CHECK(server_side_remote_endpoint.port() == client_endpoint.port());
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("ipv4/tcp");
  test->add(BOOST_TEST_CASE(&ipv4_tcp_acceptor_test));
  return test;
}
