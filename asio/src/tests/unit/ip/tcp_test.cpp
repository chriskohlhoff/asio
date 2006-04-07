//
// tcp_test.cpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/ip/tcp.hpp"

#include <boost/bind.hpp>
#include <cstring>
#include "asio.hpp"
#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// ip_tcp_socket_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ip::tcp::socket compile and link correctly. Runtime failures are ignored.

namespace ip_tcp_socket_compile {

using namespace asio;

void error_handler(const error&)
{
}

void connect_handler(const error&)
{
}

void send_handler(const error&, std::size_t)
{
}

void receive_handler(const error&, std::size_t)
{
}

void write_some_handler(const error&, std::size_t)
{
}

void read_some_handler(const error&, std::size_t)
{
}

void test()
{
  try
  {
    io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    socket_base::message_flags in_flags = 0;
    socket_base::keep_alive socket_option;
    socket_base::bytes_readable io_control_command;

    // basic_stream_socket constructors.

    ip::tcp::socket socket1(ios);
    ip::tcp::socket socket2(ios, ipv4::tcp());
    ip::tcp::socket socket3(ios, ipv6::tcp());
    ip::tcp::socket socket4(ios, ipv4::tcp::endpoint(0));
    ip::tcp::socket socket5(ios, ipv6::tcp::endpoint(0));
    int native_socket1 = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ip::tcp::socket socket6(ios, native_socket1);

    // basic_io_object functions.

    io_service& ios_ref = socket1.io_service();
    (void)ios_ref;

    // basic_socket functions.

    ip::tcp::socket::lowest_layer_type& lowest_layer = socket1.lowest_layer();
    (void)lowest_layer;

    socket1.open(ipv4::tcp());
    socket1.open(ipv6::tcp());
    socket1.open(ipv4::tcp(), error_handler);
    socket1.open(ipv6::tcp(), error_handler);
    int native_socket2 = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    socket1.open(native_socket2);
    int native_socket3 = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    socket1.open(native_socket3, error_handler);

    socket1.close();
    socket1.close(error_handler);

    ip::tcp::socket::native_type native_socket4 = socket1.native();
    (void)native_socket4;

    socket1.bind(ipv4::tcp::endpoint(0));
    socket1.bind(ipv6::tcp::endpoint(0));
    socket1.bind(ipv4::tcp::endpoint(0), error_handler);
    socket1.bind(ipv6::tcp::endpoint(0), error_handler);

    socket1.connect(ipv4::tcp::endpoint(0));
    socket1.connect(ipv6::tcp::endpoint(0));
    socket1.connect(ipv4::tcp::endpoint(0), error_handler);
    socket1.connect(ipv6::tcp::endpoint(0), error_handler);

    socket1.async_connect(ipv4::tcp::endpoint(0), connect_handler);
    socket1.async_connect(ipv6::tcp::endpoint(0), connect_handler);

    socket1.set_option(socket_option);
    socket1.set_option(socket_option, error_handler);

    socket1.get_option(socket_option);
    socket1.get_option(socket_option, error_handler);

    socket1.io_control(io_control_command);
    socket1.io_control(io_control_command, error_handler);

    ip::tcp::endpoint endpoint1 = socket1.local_endpoint();
    ip::tcp::endpoint endpoint2 = socket1.local_endpoint(error_handler);

    ip::tcp::endpoint endpoint3 = socket1.remote_endpoint();
    ip::tcp::endpoint endpoint4 = socket1.remote_endpoint(error_handler);

    socket1.shutdown(socket_base::shutdown_both);
    socket1.shutdown(socket_base::shutdown_both, error_handler);

    // basic_stream_socket functions.

    socket1.send(buffer(mutable_char_buffer));
    socket1.send(buffer(const_char_buffer));
    socket1.send(buffer(mutable_char_buffer), in_flags);
    socket1.send(buffer(const_char_buffer), in_flags);
    socket1.send(buffer(mutable_char_buffer), in_flags, error_handler);
    socket1.send(buffer(const_char_buffer), in_flags, error_handler);

    socket1.async_send(buffer(mutable_char_buffer), send_handler);
    socket1.async_send(buffer(const_char_buffer), send_handler);
    socket1.async_send(buffer(mutable_char_buffer), in_flags, send_handler);
    socket1.async_send(buffer(const_char_buffer), in_flags, send_handler);

    socket1.receive(buffer(mutable_char_buffer));
    socket1.receive(buffer(mutable_char_buffer), in_flags);
    socket1.receive(buffer(mutable_char_buffer), in_flags, error_handler);

    socket1.async_receive(buffer(mutable_char_buffer), receive_handler);
    socket1.async_receive(buffer(mutable_char_buffer), in_flags,
        receive_handler);

    socket1.write_some(buffer(mutable_char_buffer));
    socket1.write_some(buffer(const_char_buffer));
    socket1.write_some(buffer(mutable_char_buffer), error_handler);
    socket1.write_some(buffer(const_char_buffer), error_handler);

    socket1.async_write_some(buffer(mutable_char_buffer), write_some_handler);
    socket1.async_write_some(buffer(const_char_buffer), write_some_handler);

    socket1.read_some(buffer(mutable_char_buffer));
    socket1.read_some(buffer(mutable_char_buffer), error_handler);

    socket1.async_read_some(buffer(mutable_char_buffer), read_some_handler);

    socket1.peek(buffer(mutable_char_buffer));
    socket1.peek(buffer(mutable_char_buffer), error_handler);

    std::size_t in_avail1 = socket1.in_avail();
    (void)in_avail1;
    std::size_t in_avail2 = socket1.in_avail(error_handler);
    (void)in_avail2;
  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_tcp_socket_compile

//------------------------------------------------------------------------------

// ip_tcp_acceptor_runtime test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks the runtime operation of the ip::tcp::acceptor
// class.

namespace ip_tcp_acceptor_runtime {

using namespace asio;

void handle_accept(const error& err)
{
  BOOST_CHECK(!err);
}

void handle_connect(const error& err)
{
  BOOST_CHECK(!err);
}

void test()
{
  io_service ios;

  ip::tcp::acceptor acceptor(ios, ipv4::tcp::endpoint(0));
  ip::tcp::endpoint server_endpoint = acceptor.local_endpoint();
  server_endpoint.address(ipv4::address::loopback());

  ip::tcp::socket client_side_socket(ios);
  ip::tcp::socket server_side_socket(ios);

  client_side_socket.connect(server_endpoint);
  acceptor.accept(server_side_socket);

  client_side_socket.close();
  server_side_socket.close();

  client_side_socket.connect(server_endpoint);
  ip::tcp::endpoint client_endpoint;
  acceptor.accept_endpoint(server_side_socket, client_endpoint);

  ip::tcp::endpoint client_side_local_endpoint
    = client_side_socket.local_endpoint();
  BOOST_CHECK(client_side_local_endpoint.port() == client_endpoint.port());

  ip::tcp::endpoint server_side_remote_endpoint
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

} // namespace ip_tcp_acceptor_runtime

//------------------------------------------------------------------------------

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("ip/tcp");
  test->add(BOOST_TEST_CASE(&ip_tcp_socket_compile::test));
  test->add(BOOST_TEST_CASE(&ip_tcp_acceptor_runtime::test));
  return test;
}
