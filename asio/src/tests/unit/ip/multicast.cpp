//
// multicast.cpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/ip/multicast.hpp"

#include "asio.hpp"
#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// ip_multicast_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all nested classes, enums and constants in
// ip::multicast compile and link correctly. Runtime failures are ignored.

namespace ip_multicast_compile {

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  try
  {
    io_service ios;
    ip::udp::socket sock(ios);
    const ip::address address;
    const ip::address_v4 address_v4;
    const ip::address_v6 address_v6;

    // join_group class.

    ip::multicast::join_group join_group1;
    ip::multicast::join_group join_group2(address);
    ip::multicast::join_group join_group3(address_v4);
    ip::multicast::join_group join_group4(address_v4, address_v4);
    ip::multicast::join_group join_group5(address_v6);
    ip::multicast::join_group join_group6(address_v6, 1);
    sock.set_option(join_group6);

    // leave_group class.

    ip::multicast::leave_group leave_group1;
    ip::multicast::leave_group leave_group2(address);
    ip::multicast::leave_group leave_group3(address_v4);
    ip::multicast::leave_group leave_group4(address_v4, address_v4);
    ip::multicast::leave_group leave_group5(address_v6);
    ip::multicast::leave_group leave_group6(address_v6, 1);
    sock.set_option(leave_group6);

    // outbound_interface class.

    ip::multicast::outbound_interface outbound_interface1;
    ip::multicast::outbound_interface outbound_interface2(address_v4);
    ip::multicast::outbound_interface outbound_interface3(1);
    sock.set_option(outbound_interface3);

    // hops class.

    ip::multicast::hops hops1(1024);
    sock.set_option(hops1);
    ip::multicast::hops hops2;
    sock.get_option(hops2);
    hops1 = 1;
    static_cast<int>(hops1.value());

    // enable_loopback class.

    ip::multicast::enable_loopback enable_loopback1(true);
    sock.set_option(enable_loopback1);
    ip::multicast::enable_loopback enable_loopback2;
    sock.get_option(enable_loopback2);
    enable_loopback1 = true;
    static_cast<bool>(enable_loopback1);
    static_cast<bool>(!enable_loopback1);
    static_cast<bool>(enable_loopback1.value());
  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_multicast_compile

//------------------------------------------------------------------------------

// ip_multicast_runtime test
// ~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks the runtime operation of the socket options defined
// in the ip::multicast namespace.

namespace ip_multicast_runtime {

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  io_service ios;
  asio::error_code ec;

  ip::udp::endpoint ep_v4(ip::address_v4::loopback(), 0);
  ip::udp::socket sock_v4(ios);
  sock_v4.open(ep_v4.protocol(), ec);
  sock_v4.bind(ep_v4, ec);
  bool have_v4 = !ec;

  ip::udp::endpoint ep_v6(ip::address_v6::loopback(), 0);
  ip::udp::socket sock_v6(ios);
  sock_v6.open(ep_v6.protocol(), ec);
  sock_v6.bind(ep_v6, ec);
  bool have_v6 = !ec;

  BOOST_CHECK(have_v4 || have_v6);

  const ip::address multicast_address_v4 =
    ip::address::from_string("239.255.0.1", ec);
  BOOST_CHECK(!have_v4 || !ec);

  const ip::address multicast_address_v6 =
    ip::address::from_string("ff01::1", ec);
  BOOST_CHECK(!have_v6 || !ec);

  // join_group class.

  if (have_v4)
  {
    ip::multicast::join_group join_group(multicast_address_v4);
    sock_v4.set_option(join_group, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
  }

  if (have_v6)
  {
    ip::multicast::join_group join_group(multicast_address_v6);
    sock_v6.set_option(join_group, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
  }

  // leave_group class.

  if (have_v4)
  {
    ip::multicast::leave_group leave_group(multicast_address_v4);
    sock_v4.set_option(leave_group, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
  }

  if (have_v6)
  {
    ip::multicast::leave_group leave_group(multicast_address_v6);
    sock_v6.set_option(leave_group, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
  }

  // outbound_interface class.

  if (have_v4)
  {
    ip::multicast::outbound_interface outbound_interface(
        ip::address_v4::loopback());
    sock_v4.set_option(outbound_interface, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
  }

  if (have_v6)
  {
#if defined(__hpux)
    ip::multicast::outbound_interface outbound_interface(if_nametoindex("lo0"));
#else
    ip::multicast::outbound_interface outbound_interface(1);
#endif
    sock_v6.set_option(outbound_interface, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
  }

  // hops class.

  if (have_v4)
  {
    ip::multicast::hops hops1(1);
    BOOST_CHECK(hops1.value() == 1);
    sock_v4.set_option(hops1, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());

    ip::multicast::hops hops2;
    sock_v4.get_option(hops2, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
    BOOST_CHECK(hops2.value() == 1);

    ip::multicast::hops hops3(0);
    BOOST_CHECK(hops3.value() == 0);
    sock_v4.set_option(hops3, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());

    ip::multicast::hops hops4;
    sock_v4.get_option(hops4, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
    BOOST_CHECK(hops4.value() == 0);
  }

  if (have_v6)
  {
    ip::multicast::hops hops1(1);
    BOOST_CHECK(hops1.value() == 1);
    sock_v6.set_option(hops1, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());

    ip::multicast::hops hops2;
    sock_v6.get_option(hops2, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
    BOOST_CHECK(hops2.value() == 1);

    ip::multicast::hops hops3(0);
    BOOST_CHECK(hops3.value() == 0);
    sock_v6.set_option(hops3, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());

    ip::multicast::hops hops4;
    sock_v6.get_option(hops4, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
    BOOST_CHECK(hops4.value() == 0);
  }

  // enable_loopback class.

  if (have_v4)
  {
    ip::multicast::enable_loopback enable_loopback1(true);
    BOOST_CHECK(enable_loopback1.value());
    BOOST_CHECK(static_cast<bool>(enable_loopback1));
    BOOST_CHECK(!!enable_loopback1);
    sock_v4.set_option(enable_loopback1, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());

    ip::multicast::enable_loopback enable_loopback2;
    sock_v4.get_option(enable_loopback2, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
    BOOST_CHECK(enable_loopback2.value());
    BOOST_CHECK(static_cast<bool>(enable_loopback2));
    BOOST_CHECK(!!enable_loopback2);

    ip::multicast::enable_loopback enable_loopback3(false);
    BOOST_CHECK(!enable_loopback3.value());
    BOOST_CHECK(!static_cast<bool>(enable_loopback3));
    BOOST_CHECK(!enable_loopback3);
    sock_v4.set_option(enable_loopback3, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());

    ip::multicast::enable_loopback enable_loopback4;
    sock_v4.get_option(enable_loopback4, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
    BOOST_CHECK(!enable_loopback4.value());
    BOOST_CHECK(!static_cast<bool>(enable_loopback4));
    BOOST_CHECK(!enable_loopback4);
  }

  if (have_v6)
  {
    ip::multicast::enable_loopback enable_loopback1(true);
    BOOST_CHECK(enable_loopback1.value());
    BOOST_CHECK(static_cast<bool>(enable_loopback1));
    BOOST_CHECK(!!enable_loopback1);
    sock_v6.set_option(enable_loopback1, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());

    ip::multicast::enable_loopback enable_loopback2;
    sock_v6.get_option(enable_loopback2, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
    BOOST_CHECK(enable_loopback2.value());
    BOOST_CHECK(static_cast<bool>(enable_loopback2));
    BOOST_CHECK(!!enable_loopback2);

    ip::multicast::enable_loopback enable_loopback3(false);
    BOOST_CHECK(!enable_loopback3.value());
    BOOST_CHECK(!static_cast<bool>(enable_loopback3));
    BOOST_CHECK(!enable_loopback3);
    sock_v6.set_option(enable_loopback3, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());

    ip::multicast::enable_loopback enable_loopback4;
    sock_v6.get_option(enable_loopback4, ec);
    BOOST_CHECK_MESSAGE(!ec, ec.value() << ", " << ec.message());
    BOOST_CHECK(!enable_loopback4.value());
    BOOST_CHECK(!static_cast<bool>(enable_loopback4));
    BOOST_CHECK(!enable_loopback4);
  }
}

} // namespace ip_multicast_runtime

//------------------------------------------------------------------------------

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("ip/multicast");
  test->add(BOOST_TEST_CASE(&ip_multicast_compile::test));
  test->add(BOOST_TEST_CASE(&ip_multicast_runtime::test));
  return test;
}
