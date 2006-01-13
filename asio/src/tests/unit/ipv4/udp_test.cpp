//
// udp_test.cpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/ipv4/udp.hpp"

#include <boost/bind.hpp>
#include <cstring>
#include "asio.hpp"
#include "../unit_test.hpp"

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

void ipv4_udp_test()
{
  using namespace std; // For memcmp and memset.

  io_service ios;

  ipv4::udp::socket s1(ios, ipv4::udp::endpoint(0));
  ipv4::udp::endpoint target_endpoint;
  s1.get_local_endpoint(target_endpoint);
  target_endpoint.address(ipv4::address::loopback());

  ipv4::udp::socket s2(ios);
  s2.open(ipv4::udp());
  s2.bind(ipv4::udp::endpoint(0));
  char send_msg[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  s2.send_to(buffer(send_msg, sizeof(send_msg)), 0, target_endpoint);

  char recv_msg[sizeof(send_msg)];
  ipv4::udp::endpoint sender_endpoint;
  size_t bytes_recvd = s1.receive_from(buffer(recv_msg, sizeof(recv_msg)), 0,
      sender_endpoint);

  BOOST_CHECK(bytes_recvd == sizeof(send_msg));
  BOOST_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);

  memset(recv_msg, 0, sizeof(recv_msg));

  target_endpoint = sender_endpoint;
  s1.async_send_to(buffer(send_msg, sizeof(send_msg)), 0, target_endpoint,
      boost::bind(handle_send, sizeof(send_msg),
        placeholders::error, placeholders::bytes_transferred));
  s2.async_receive_from(buffer(recv_msg, sizeof(recv_msg)), 0, sender_endpoint,
      boost::bind(handle_recv, sizeof(recv_msg),
        placeholders::error, placeholders::bytes_transferred));

  ios.run();

  BOOST_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("ipv4/udp");
  test->add(BOOST_TEST_CASE(&ipv4_udp_test));
  return test;
}
