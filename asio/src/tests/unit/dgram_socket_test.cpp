//
// dgram_socket_test.hpp
// ~~~~~~~~~~~~~~
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

void dgram_socket_test()
{
  using namespace std; // For memcmp.

  demuxer d;

  dgram_socket s1(d, inet_address_v4(0));
  inet_address_v4 target_addr;
  s1.get_local_address(target_addr);

  dgram_socket s2(d, inet_address_v4(0));
  char send_msg[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  s2.sendto(send_msg, sizeof(send_msg), target_addr);

  char recv_msg[sizeof(send_msg)];
  inet_address_v4 sender_addr;
  size_t bytes_recvd = s1.recvfrom(recv_msg, sizeof(recv_msg), sender_addr);

  UNIT_TEST_CHECK(bytes_recvd == sizeof(send_msg));
  UNIT_TEST_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);
}

UNIT_TEST(dgram_socket_test)
