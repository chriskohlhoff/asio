//
// dgram_socket_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~
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

void handle_send(size_t expected_bytes_sent, const socket_error& error,
    size_t bytes_sent)
{
  UNIT_TEST_CHECK(!error);
  UNIT_TEST_CHECK(expected_bytes_sent == bytes_sent);
}

void handle_recv(size_t expected_bytes_recvd, const socket_error& error,
    size_t bytes_recvd)
{
  UNIT_TEST_CHECK(!error);
  UNIT_TEST_CHECK(expected_bytes_recvd == bytes_recvd);
}

void dgram_socket_test()
{
  using namespace std; // For memcmp and memset.

  demuxer d;

  dgram_socket s1(d, ipv4::udp::endpoint(0));
  ipv4::udp::endpoint target_endpoint;
  s1.get_local_endpoint(target_endpoint);
  target_endpoint.address(ipv4::address::loopback());

  dgram_socket s2(d);
  s2.open(ipv4::udp());
  s2.bind(ipv4::udp::endpoint(0));
  char send_msg[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  s2.sendto(send_msg, sizeof(send_msg), target_endpoint);

  char recv_msg[sizeof(send_msg)];
  ipv4::udp::endpoint sender_endpoint;
  size_t bytes_recvd = s1.recvfrom(recv_msg, sizeof(recv_msg),
      sender_endpoint);

  UNIT_TEST_CHECK(bytes_recvd == sizeof(send_msg));
  UNIT_TEST_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);

  memset(recv_msg, 0, sizeof(recv_msg));

  target_endpoint = sender_endpoint;
  s1.async_sendto(send_msg, sizeof(send_msg), target_endpoint,
      boost::bind(handle_send, sizeof(send_msg), arg::error, arg::bytes_sent));
  s2.async_recvfrom(recv_msg, sizeof(recv_msg), sender_endpoint,
      boost::bind(handle_recv, sizeof(recv_msg), arg::error, arg::bytes_recvd));

  d.run();

  UNIT_TEST_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);
}

UNIT_TEST(dgram_socket_test)
