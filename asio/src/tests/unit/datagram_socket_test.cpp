//
// datagram_socket_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <boost/bind.hpp>
#include <cstring>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void handle_send(size_t expected_bytes_sent, const error& err,
    size_t bytes_sent)
{
  UNIT_TEST_CHECK(!err);
  UNIT_TEST_CHECK(expected_bytes_sent == bytes_sent);
}

void handle_recv(size_t expected_bytes_recvd, const error& err,
    size_t bytes_recvd)
{
  UNIT_TEST_CHECK(!err);
  UNIT_TEST_CHECK(expected_bytes_recvd == bytes_recvd);
}

void datagram_socket_test()
{
  using namespace std; // For memcmp and memset.

  demuxer d;

  datagram_socket s1(d, ipv4::udp::endpoint(0));
  ipv4::udp::endpoint target_endpoint;
  s1.get_local_endpoint(target_endpoint);
  target_endpoint.address(ipv4::address::loopback());

  datagram_socket s2(d);
  s2.open(ipv4::udp());
  s2.bind(ipv4::udp::endpoint(0));
  char send_msg[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  s2.send_to(send_msg, sizeof(send_msg), target_endpoint);

  char recv_msg[sizeof(send_msg)];
  ipv4::udp::endpoint sender_endpoint;
  size_t bytes_recvd = s1.receive_from(recv_msg, sizeof(recv_msg),
      sender_endpoint);

  UNIT_TEST_CHECK(bytes_recvd == sizeof(send_msg));
  UNIT_TEST_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);

  memset(recv_msg, 0, sizeof(recv_msg));

  target_endpoint = sender_endpoint;
  s1.async_send_to(send_msg, sizeof(send_msg), target_endpoint,
      boost::bind(handle_send, sizeof(send_msg),
        placeholders::error, placeholders::bytes_transferred));
  s2.async_receive_from(recv_msg, sizeof(recv_msg), sender_endpoint,
      boost::bind(handle_recv, sizeof(recv_msg),
        placeholders::error, placeholders::bytes_transferred));

  d.run();

  UNIT_TEST_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);
}

UNIT_TEST(datagram_socket_test)
