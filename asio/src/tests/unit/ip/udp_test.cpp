//
// udp_test.cpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/ip/udp.hpp"

#include <boost/bind.hpp>
#include <cstring>
#include "asio.hpp"
#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// ip_udp_socket_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ip::udp::socket compile and link correctly. Runtime failures are ignored.

namespace ip_udp_socket_compile {

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

    // basic_datagram_socket constructors.

    ip::udp::socket socket1(ios);
    ip::udp::socket socket2(ios, ipv4::udp());
    ip::udp::socket socket3(ios, ipv6::udp());
    ip::udp::socket socket4(ios, ipv4::udp::endpoint(0));
    ip::udp::socket socket5(ios, ipv6::udp::endpoint(0));
    int native_socket1 = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    ip::udp::socket socket6(ios, ipv4::udp(), native_socket1);

    // basic_io_object functions.

    io_service& ios_ref = socket1.io_service();
    (void)ios_ref;

    // basic_socket functions.

    ip::udp::socket::lowest_layer_type& lowest_layer = socket1.lowest_layer();
    (void)lowest_layer;

    socket1.open(ipv4::udp());
    socket1.open(ipv6::udp());
    socket1.open(ipv4::udp(), error_handler);
    socket1.open(ipv6::udp(), error_handler);

    int native_socket2 = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    socket1.assign(ipv4::udp(), native_socket2);
    int native_socket3 = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    socket1.assign(ipv4::udp(), native_socket3, error_handler);

    socket1.close();
    socket1.close(error_handler);

    ip::udp::socket::native_type native_socket4 = socket1.native();
    (void)native_socket4;

    socket1.bind(ipv4::udp::endpoint(0));
    socket1.bind(ipv6::udp::endpoint(0));
    socket1.bind(ipv4::udp::endpoint(0), error_handler);
    socket1.bind(ipv6::udp::endpoint(0), error_handler);

    socket1.connect(ipv4::udp::endpoint(0));
    socket1.connect(ipv6::udp::endpoint(0));
    socket1.connect(ipv4::udp::endpoint(0), error_handler);
    socket1.connect(ipv6::udp::endpoint(0), error_handler);

    socket1.async_connect(ipv4::udp::endpoint(0), connect_handler);
    socket1.async_connect(ipv6::udp::endpoint(0), connect_handler);

    socket1.set_option(socket_option);
    socket1.set_option(socket_option, error_handler);

    socket1.get_option(socket_option);
    socket1.get_option(socket_option, error_handler);

    socket1.io_control(io_control_command);
    socket1.io_control(io_control_command, error_handler);

    ip::udp::endpoint endpoint1 = socket1.local_endpoint();
    ip::udp::endpoint endpoint2 = socket1.local_endpoint(error_handler);

    ip::udp::endpoint endpoint3 = socket1.remote_endpoint();
    ip::udp::endpoint endpoint4 = socket1.remote_endpoint(error_handler);

    socket1.shutdown(socket_base::shutdown_both);
    socket1.shutdown(socket_base::shutdown_both, error_handler);

    // basic_datagram_socket functions.

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

    socket1.send_to(buffer(mutable_char_buffer), ipv4::udp::endpoint(0));
    socket1.send_to(buffer(mutable_char_buffer), ipv6::udp::endpoint(0));
    socket1.send_to(buffer(const_char_buffer), ipv4::udp::endpoint(0));
    socket1.send_to(buffer(const_char_buffer), ipv6::udp::endpoint(0));
    socket1.send_to(buffer(mutable_char_buffer),
        ipv4::udp::endpoint(0), in_flags);
    socket1.send_to(buffer(mutable_char_buffer),
        ipv6::udp::endpoint(0), in_flags);
    socket1.send_to(buffer(const_char_buffer),
        ipv4::udp::endpoint(0), in_flags);
    socket1.send_to(buffer(const_char_buffer),
        ipv6::udp::endpoint(0), in_flags);
    socket1.send_to(buffer(mutable_char_buffer),
        ipv4::udp::endpoint(0), in_flags, error_handler);
    socket1.send_to(buffer(mutable_char_buffer),
        ipv6::udp::endpoint(0), in_flags, error_handler);
    socket1.send_to(buffer(const_char_buffer),
        ipv4::udp::endpoint(0), in_flags, error_handler);
    socket1.send_to(buffer(const_char_buffer),
        ipv6::udp::endpoint(0), in_flags, error_handler);

    socket1.async_send_to(buffer(mutable_char_buffer),
        ipv4::udp::endpoint(0), send_handler);
    socket1.async_send_to(buffer(mutable_char_buffer),
        ipv6::udp::endpoint(0), send_handler);
    socket1.async_send_to(buffer(const_char_buffer),
        ipv4::udp::endpoint(0), send_handler);
    socket1.async_send_to(buffer(const_char_buffer),
        ipv6::udp::endpoint(0), send_handler);
    socket1.async_send_to(buffer(mutable_char_buffer),
        ipv4::udp::endpoint(0), in_flags, send_handler);
    socket1.async_send_to(buffer(mutable_char_buffer),
        ipv6::udp::endpoint(0), in_flags, send_handler);
    socket1.async_send_to(buffer(const_char_buffer),
        ipv4::udp::endpoint(0), in_flags, send_handler);
    socket1.async_send_to(buffer(const_char_buffer),
        ipv6::udp::endpoint(0), in_flags, send_handler);

    socket1.receive(buffer(mutable_char_buffer));
    socket1.receive(buffer(mutable_char_buffer), in_flags);
    socket1.receive(buffer(mutable_char_buffer), in_flags, error_handler);

    socket1.async_receive(buffer(mutable_char_buffer), receive_handler);
    socket1.async_receive(buffer(mutable_char_buffer), in_flags,
        receive_handler);

    ip::udp::endpoint endpoint;
    socket1.receive_from(buffer(mutable_char_buffer), endpoint);
    socket1.receive_from(buffer(mutable_char_buffer), endpoint, in_flags);
    socket1.receive_from(buffer(mutable_char_buffer),
        endpoint, in_flags, error_handler);

    socket1.async_receive_from(buffer(mutable_char_buffer),
        endpoint, receive_handler);
    socket1.async_receive_from(buffer(mutable_char_buffer),
        endpoint, in_flags, receive_handler);
  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_udp_socket_compile

//------------------------------------------------------------------------------

// ip_udp_socket_runtime test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks the runtime operation of the ip::udp::socket class.

namespace ip_udp_socket_runtime {

using namespace asio;

void handle_send(size_t expected_bytes_sent, const error& err,
    size_t bytes_sent)
{
  BOOST_CHECK(!err);
  BOOST_CHECK(expected_bytes_sent == bytes_sent);
}

void handle_recv(size_t expected_bytes_recvd, const error& err,
    size_t bytes_recvd)
{
  BOOST_CHECK(!err);
  BOOST_CHECK(expected_bytes_recvd == bytes_recvd);
}

void test()
{
  using namespace std; // For memcmp and memset.

  io_service ios;

  ip::udp::socket s1(ios, ipv4::udp::endpoint(0));
  ip::udp::endpoint target_endpoint = s1.local_endpoint();
  target_endpoint.address(ipv4::address::loopback());

  ip::udp::socket s2(ios);
  s2.open(ipv4::udp());
  s2.bind(ipv4::udp::endpoint(0));
  char send_msg[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  s2.send_to(buffer(send_msg, sizeof(send_msg)), target_endpoint);

  char recv_msg[sizeof(send_msg)];
  ip::udp::endpoint sender_endpoint;
  size_t bytes_recvd = s1.receive_from(buffer(recv_msg, sizeof(recv_msg)),
      sender_endpoint);

  BOOST_CHECK(bytes_recvd == sizeof(send_msg));
  BOOST_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);

  memset(recv_msg, 0, sizeof(recv_msg));

  target_endpoint = sender_endpoint;
  s1.async_send_to(buffer(send_msg, sizeof(send_msg)), target_endpoint,
      boost::bind(handle_send, sizeof(send_msg),
        placeholders::error, placeholders::bytes_transferred));
  s2.async_receive_from(buffer(recv_msg, sizeof(recv_msg)), sender_endpoint,
      boost::bind(handle_recv, sizeof(recv_msg),
        placeholders::error, placeholders::bytes_transferred));

  ios.run();

  BOOST_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);
}

} // namespace ip_udp_socket_runtime

//------------------------------------------------------------------------------

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("ip/udp");
  test->add(BOOST_TEST_CASE(&ip_udp_socket_compile::test));
  test->add(BOOST_TEST_CASE(&ip_udp_socket_runtime::test));
  return test;
}
