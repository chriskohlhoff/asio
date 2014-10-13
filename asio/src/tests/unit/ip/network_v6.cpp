//
// network_v6.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/ip/network_v6.hpp"

#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// ip_network_v6_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ip::network_v6 compile and link correctly. Runtime failures are ignored.

namespace ip_network_v6_compile {

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  try
  {
    asio::error_code ec;

    // network_v6 constructors.

    ip::network_v6 net1(ip::make_address_v6("2001:370::10:7344"), 64);
    ip::network_v6 net2(ip::make_address_v6("2001:370::10:7344"),
        ip::make_address_v6("2001:0370:"));

    // network_v6 functions.

    ip::address_v6 addr1 = net1.address();
    (void)addr1;

    unsigned short prefix_len = net1.prefix_length();
    (void)prefix_len;

    ip::address_v6 addr2 = net1.netmask();
    (void)addr2;

    ip::address_v6 addr3 = net1.network();
    (void)addr3;

    ip::address_range_v6 hosts = net1.hosts();
    (void)hosts;

    ip::network_v6 net3 = net1.canonical();
    (void)net3;

    bool b1 = net1.is_host();
    (void)b1;

    bool b2 = net1.is_subnet_of(net2);
    (void)b2;

    std::string s1 = net1.to_string();
    (void)s1;

    std::string s2 = net1.to_string(ec);
    (void)s2;

    // network_v6 comparisons.

    bool b3 = (net1 == net2);
    (void)b3;

    bool b4 = (net1 != net2);
    (void)b4;

  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_network_v6_compile

//------------------------------------------------------------------------------

// ip_network_v6_runtime test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that the various public member functions meet the
// necessary postconditions.

namespace ip_network_v6_runtime {

void test()
{
  using asio::ip::address_v6;
  using asio::ip::make_address_v6;
  using asio::ip::network_v6;
  using asio::ip::make_network_v6;

  address_v6 addr = make_address_v6("2001:370::10:7344");

  // calculate prefix length

  network_v6 net1(addr,
    make_address_v6("ffff:ffff:ffff:ffff:0000:0000:0000:0000"));
  ASIO_CHECK(net1.prefix_length() == 64);

  network_v6 net2(addr,
    make_address_v6("ffff:ffff:ffff:ffff:fc00:0000:0000:0000"));
  ASIO_CHECK(net2.prefix_length() == 70);

  network_v6 net3(addr,
    make_address_v6("8000:0000:0000:0000:0000:0000:0000:0000"));
  ASIO_CHECK(net3.prefix_length() == 1);

  std::string msg;
  try
  {
    make_network_v6(addr, make_address_v6("0000:ffff::"));
  }
  catch(std::exception& ex)
  {
    msg = ex.what();
  }
  ASIO_CHECK(msg == std::string("non-contiguous netmask"));

  // calculate netmask
  network_v6 net4(addr, 64);
  ASIO_CHECK(net4.netmask() == make_address_v6("ffff:ffff:ffff:ffff:0000:0000:0000:0000"));

  network_v6 net5(addr, 63);
  ASIO_CHECK(net5.netmask() == make_address_v6("ffff:ffff:ffff:fffe:0000:0000:0000:0000"));

  network_v6 net6(addr, 24);
  ASIO_CHECK(net6.netmask() == make_address_v6("ffff:ff00:0000:0000:0000:0000:0000:0000"));

  network_v6 net7(addr, 16);
  ASIO_CHECK(net7.netmask() == make_address_v6("ffff:0000:0000:0000:0000:0000:0000:0000"));

  network_v6 net8(addr, 8);
  ASIO_CHECK(net8.netmask() == make_address_v6("ff00:0000:0000:0000:0000:0000:0000:0000"));

  network_v6 net9(addr, 128);
  ASIO_CHECK(net9.netmask() == make_address_v6("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"));

  network_v6 net10(addr, 1);
  ASIO_CHECK(net10.netmask() == make_address_v6("8000:0000:0000:0000:0000:0000:0000:0000"));

  network_v6 net11(addr, 0);
  ASIO_CHECK(net11.netmask() == make_address_v6("::"));

  msg.clear();
  try
  {
    make_network_v6(addr, 129);
  }
  catch(std::out_of_range& ex)
  {
    msg = ex.what();
  }
  ASIO_CHECK(msg == std::string("prefix length too large"));

#if 0
  // construct address range from address and prefix length
  ASIO_CHECK(network_v6(address_v6::from_string("192.168.77.100"), 32).network() == address_v6::from_string("192.168.77.100"));
  ASIO_CHECK(network_v6(address_v6::from_string("192.168.77.100"), 24).network() == address_v6::from_string("192.168.77.0"));
  ASIO_CHECK(network_v6(address_v6::from_string("192.168.77.128"), 25).network() == address_v6::from_string("192.168.77.128"));

  // construct address range from string in CIDR notation
  ASIO_CHECK(make_network_v6("192.168.77.100/32").network() == address_v6::from_string("192.168.77.100"));
  ASIO_CHECK(make_network_v6("192.168.77.100/24").network() == address_v6::from_string("192.168.77.0"));
  ASIO_CHECK(make_network_v6("192.168.77.128/25").network() == address_v6::from_string("192.168.77.128"));

  // prefix length
  ASIO_CHECK(make_network_v6("193.99.144.80/24").prefix_length() == 24);
  ASIO_CHECK(network_v6(address_v6::from_string("193.99.144.80"), 24).prefix_length() == 24);
  ASIO_CHECK(network_v6(address_v6::from_string("192.168.77.0"), address_v6::from_string("255.255.255.0")).prefix_length() == 24);

  // to string
  std::string a("192.168.77.0/32");
  ASIO_CHECK(make_network_v6(a.c_str()).to_string() == a);
  ASIO_CHECK(network_v6(address_v6::from_string("192.168.77.10"), 24).to_string() == std::string("192.168.77.10/24"));

  // return host part
  ASIO_CHECK(make_network_v6("192.168.77.11/24").address() == address_v6::from_string("192.168.77.11"));

  // return host in CIDR notation
  ASIO_CHECK(make_network_v6("192.168.78.30/20").address().to_string() == "192.168.78.30");

  // return network in CIDR notation
  ASIO_CHECK(make_network_v6("192.168.78.30/20").canonical().to_string() == "192.168.64.0/20");

  // is host
  ASIO_CHECK(make_network_v6("192.168.77.0/32").is_host());
  ASIO_CHECK(!make_network_v6("192.168.77.0/31").is_host());

  // is real subnet of
  ASIO_CHECK(make_network_v6("192.168.0.192/24").is_subnet_of(make_network_v6("192.168.0.0/16")));
  ASIO_CHECK(make_network_v6("192.168.0.0/24").is_subnet_of(make_network_v6("192.168.192.168/16")));
  ASIO_CHECK(make_network_v6("192.168.0.192/24").is_subnet_of(make_network_v6("192.168.192.168/16")));
  ASIO_CHECK(make_network_v6("192.168.0.0/24").is_subnet_of(make_network_v6("192.168.0.0/16")));
  ASIO_CHECK(make_network_v6("192.168.0.0/24").is_subnet_of(make_network_v6("192.168.0.0/23")));
  ASIO_CHECK(make_network_v6("192.168.0.0/24").is_subnet_of(make_network_v6("192.168.0.0/0")));
  ASIO_CHECK(make_network_v6("192.168.0.0/32").is_subnet_of(make_network_v6("192.168.0.0/24")));

  ASIO_CHECK(!make_network_v6("192.168.0.0/32").is_subnet_of(make_network_v6("192.168.0.0/32")));
  ASIO_CHECK(!make_network_v6("192.168.0.0/24").is_subnet_of(make_network_v6("192.168.1.0/24")));
  ASIO_CHECK(!make_network_v6("192.168.0.0/16").is_subnet_of(make_network_v6("192.168.1.0/24")));

  network_v6 r(make_network_v6("192.168.0.0/24"));
  ASIO_CHECK(!r.is_subnet_of(r));

  network_v6 net12(make_network_v6("192.168.0.2/24"));
  network_v6 net13(make_network_v6("192.168.1.1/28"));
  network_v6 net14(make_network_v6("192.168.1.21/28"));
  // network
  ASIO_CHECK(net12.network() == address_v6::from_string("192.168.0.0"));
  ASIO_CHECK(net13.network() == address_v6::from_string("192.168.1.0"));
  ASIO_CHECK(net14.network() == address_v6::from_string("192.168.1.16"));
  // netmask
  ASIO_CHECK(net12.netmask() == address_v6::from_string("255.255.255.0"));
  ASIO_CHECK(net13.netmask() == address_v6::from_string("255.255.255.240"));
  ASIO_CHECK(net14.netmask() == address_v6::from_string("255.255.255.240"));
  // iterator
  ASIO_CHECK(std::distance(net12.hosts().begin(),net12.hosts().end()) == 254);
  ASIO_CHECK(*net12.hosts().begin() == address_v6::from_string("192.168.0.1"));
  ASIO_CHECK(net12.hosts().end() != net12.hosts().find(address_v6::from_string("192.168.0.10")));
  ASIO_CHECK(net12.hosts().end() == net12.hosts().find(address_v6::from_string("192.168.1.10")));
  ASIO_CHECK(std::distance(net13.hosts().begin(),net13.hosts().end()) == 14);
  ASIO_CHECK(*net13.hosts().begin() == address_v6::from_string("192.168.1.1"));
  ASIO_CHECK(net13.hosts().end() != net13.hosts().find(address_v6::from_string("192.168.1.14")));
  ASIO_CHECK(net13.hosts().end() == net13.hosts().find(address_v6::from_string("192.168.1.15")));
  ASIO_CHECK(std::distance(net14.hosts().begin(),net14.hosts().end()) == 14);
  ASIO_CHECK(*net14.hosts().begin() == address_v6::from_string("192.168.1.17"));
  ASIO_CHECK(net14.hosts().end() != net14.hosts().find(address_v6::from_string("192.168.1.30")));
  ASIO_CHECK(net14.hosts().end() == net14.hosts().find(address_v6::from_string("192.168.1.31")));
#endif
}

} // namespace ip_network_v6_runtime

//------------------------------------------------------------------------------

ASIO_TEST_SUITE
(
  "ip/network_v6",
  ASIO_TEST_CASE(ip_network_v6_compile::test)
  ASIO_TEST_CASE(ip_network_v6_runtime::test)
)
