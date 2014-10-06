//
// network_v4.cpp
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
#include "asio/ip/network_v4.hpp"

#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// ip_network_v4_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ip::network_v4 compile and link correctly. Runtime failures are ignored.

namespace ip_network_v4_compile {

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  try
  {
    asio::error_code ec;

    // network_v4 constructors.

    ip::network_v4 net1(ip::make_address_v4("192.168.1.0"), 32);
    ip::network_v4 net2(ip::make_address_v4("192.168.1.0"),
        ip::make_address_v4("255.255.255.0"));

    // network_v4 functions.

    ip::address_v4 addr1 = net1.address();
    (void)addr1;

    unsigned short prefix_len = net1.prefix_length();
    (void)prefix_len;

    ip::address_v4 addr2 = net1.netmask();
    (void)addr2;

    ip::address_v4 addr3 = net1.network();
    (void)addr3;

    ip::address_v4 addr4 = net1.broadcast();
    (void)addr4;

    ip::address_range_v4 hosts = net1.hosts();
    (void)hosts;

    ip::network_v4 net3 = net1.canonical();
    (void)net3;

    bool b1 = net1.is_host();
    (void)b1;

    bool b2 = net1.is_subnet_of(net2);
    (void)b2;

    std::string s1 = net1.to_string();
    (void)s1;

    std::string s2 = net1.to_string(ec);
    (void)s2;

    // network_v4 comparisons.

    bool b3 = (net1 == net2);
    (void)b3;

    bool b4 = (net1 != net2);
    (void)b4;

  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_network_v4_compile

//------------------------------------------------------------------------------

// ip_network_v4_runtime test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that the various public member functions meet the
// necessary postconditions.

namespace ip_network_v4_runtime {

void test()
{
  using asio::ip::address_v4;
  using asio::ip::make_address_v4;
  using asio::ip::network_v4;
  using asio::ip::make_network_v4;

  address_v4 addr = make_address_v4("1.2.3.4");

  // calculate prefix length

  network_v4 net1(addr, make_address_v4("255.255.255.0"));
  ASIO_CHECK(net1.prefix_length() == 24);

  network_v4 net2(addr, make_address_v4("255.255.255.192"));
  ASIO_CHECK(net2.prefix_length() == 26);

  network_v4 net3(addr, make_address_v4("128.0.0.0"));
  ASIO_CHECK(net3.prefix_length() == 1);

  std::string msg;
  try
  {
    make_network_v4(addr, make_address_v4("255.255.255.1"));
  }
  catch(std::exception& ex)
  {
    msg = ex.what();
  }
  ASIO_CHECK(msg == std::string("non-contiguous netmask"));

  msg.clear();
  try
  {
    make_network_v4(addr, make_address_v4("0.255.255.0"));
  }
  catch(std::exception& ex)
  {
    msg = ex.what();
  }
  ASIO_CHECK(msg == std::string("non-contiguous netmask"));

  // calculate netmask

  network_v4 net4(addr, 23);
  ASIO_CHECK(net4.netmask() == make_address_v4("255.255.254.0"));

  network_v4 net5(addr, 12);
  ASIO_CHECK(net5.netmask() == make_address_v4("255.240.0.0"));

  network_v4 net6(addr, 24);
  ASIO_CHECK(net6.netmask() == make_address_v4("255.255.255.0"));

  network_v4 net7(addr, 16);
  ASIO_CHECK(net7.netmask() == make_address_v4("255.255.0.0"));

  network_v4 net8(addr, 8);
  ASIO_CHECK(net8.netmask() == make_address_v4("255.0.0.0"));

  network_v4 net9(addr, 32);
  ASIO_CHECK(net9.netmask() == make_address_v4("255.255.255.255"));

  network_v4 net10(addr, 1);
  ASIO_CHECK(net10.netmask() == make_address_v4("128.0.0.0"));

  network_v4 net11(addr, 0);
  ASIO_CHECK(net11.netmask() == make_address_v4("0.0.0.0"));

  msg.clear();
  try
  {
    make_network_v4(addr, 33);
  }
  catch(std::out_of_range& ex)
  {
    msg = ex.what();
  }
  ASIO_CHECK(msg == std::string("prefix length too large"));

#if 0
  // construct address range from address and prefix length
  ASIO_CHECK(network_v4(address_v4::from_string("192.168.77.100"), 32).network() == address_v4::from_string("192.168.77.100"));
  ASIO_CHECK(network_v4(address_v4::from_string("192.168.77.100"), 24).network() == address_v4::from_string("192.168.77.0"));
  ASIO_CHECK(network_v4(address_v4::from_string("192.168.77.128"), 25).network() == address_v4::from_string("192.168.77.128"));

  // construct address range from string in CIDR notation
  ASIO_CHECK(network_v4::from_string("192.168.77.100/32").network() == address_v4::from_string("192.168.77.100"));
  ASIO_CHECK(network_v4::from_string("192.168.77.100/24").network() == address_v4::from_string("192.168.77.0"));
  ASIO_CHECK(network_v4::from_string("192.168.77.128/25").network() == address_v4::from_string("192.168.77.128"));

  // prefix length
  ASIO_CHECK(network_v4::from_string("193.99.144.80/24").prefix_length() == 24);
  ASIO_CHECK(network_v4(address_v4::from_string("193.99.144.80"), 24).prefix_length() == 24);
  ASIO_CHECK(network_v4(address_v4::from_string("192.168.77.0"), address_v4::from_string("255.255.255.0")).prefix_length() == 24);

  // to string
  std::string a("192.168.77.0/32");
  ASIO_CHECK(network_v4::from_string(a.c_str()).to_string() == a);
  ASIO_CHECK(network_v4(address_v4::from_string("192.168.77.10"), 24).to_string() == std::string("192.168.77.10/24"));

  // return host part
  ASIO_CHECK(network_v4::from_string("192.168.77.11/24").host() == address_v4::from_string("192.168.77.11"));

  // return host in CIDR notation
  ASIO_CHECK(network_v4::from_string("192.168.78.30/20").host_cidr().to_string() == "192.168.78.30/32");

  // return network in CIDR notation
  ASIO_CHECK(network_v4::from_string("192.168.78.30/20").network_cidr().to_string() == "192.168.64.0/20");

  // is host
  ASIO_CHECK(network_v4::from_string("192.168.77.0/32").is_host());
  ASIO_CHECK(!network_v4::from_string("192.168.77.0/31").is_host());

  // is real subnet of
  ASIO_CHECK(network_v4::from_string("192.168.0.192/24").is_subnet_of(network_v4::from_string("192.168.0.0/16")));
  ASIO_CHECK(network_v4::from_string("192.168.0.0/24").is_subnet_of(network_v4::from_string("192.168.192.168/16")));
  ASIO_CHECK(network_v4::from_string("192.168.0.192/24").is_subnet_of(network_v4::from_string("192.168.192.168/16")));
  ASIO_CHECK(network_v4::from_string("192.168.0.0/24").is_subnet_of(network_v4::from_string("192.168.0.0/16")));
  ASIO_CHECK(network_v4::from_string("192.168.0.0/24").is_subnet_of(network_v4::from_string("192.168.0.0/23")));
  ASIO_CHECK(network_v4::from_string("192.168.0.0/24").is_subnet_of(network_v4::from_string("192.168.0.0/0")));
  ASIO_CHECK(network_v4::from_string("192.168.0.0/32").is_subnet_of(network_v4::from_string("192.168.0.0/24")));

  ASIO_CHECK(!network_v4::from_string("192.168.0.0/32").is_subnet_of(network_v4::from_string("192.168.0.0/32")));
  ASIO_CHECK(!network_v4::from_string("192.168.0.0/24").is_subnet_of(network_v4::from_string("192.168.1.0/24")));
  ASIO_CHECK(!network_v4::from_string("192.168.0.0/16").is_subnet_of(network_v4::from_string("192.168.1.0/24")));

  network_v4 r(network_v4::from_string("192.168.0.0/24"));
  ASIO_CHECK(!r.is_subnet_of(r));

  network_v4 addr1(network_v4::from_string("192.168.0.2/24"));
  network_v4 addr2(network_v4::from_string("192.168.1.1/28"));
  network_v4 addr3(network_v4::from_string("192.168.1.21/28"));
  // network
  ASIO_CHECK(addr1.network() == address_v4::from_string("192.168.0.0"));
  ASIO_CHECK(addr2.network() == address_v4::from_string("192.168.1.0"));
  ASIO_CHECK(addr3.network() == address_v4::from_string("192.168.1.16"));
  // netmask
  ASIO_CHECK(addr1.netmask() == address_v4::from_string("255.255.255.0"));
  ASIO_CHECK(addr2.netmask() == address_v4::from_string("255.255.255.240"));
  ASIO_CHECK(addr3.netmask() == address_v4::from_string("255.255.255.240"));
  // broadcast
  ASIO_CHECK(addr1.broadcast() == address_v4::from_string("192.168.0.255"));
  ASIO_CHECK(addr2.broadcast() == address_v4::from_string("192.168.1.15"));
  ASIO_CHECK(addr3.broadcast() == address_v4::from_string("192.168.1.31"));
  // iterator
  ASIO_CHECK(std::distance(addr1.begin(),addr1.end()) == 254);
  ASIO_CHECK(*addr1.begin() == address_v4::from_string("192.168.0.1"));
  ASIO_CHECK(addr1.end() != addr1.find(address_v4::from_string("192.168.0.10")));
  ASIO_CHECK(addr1.end() == addr1.find(address_v4::from_string("192.168.1.10")));
  ASIO_CHECK(std::distance(addr2.begin(),addr2.end()) == 14);
  ASIO_CHECK(*addr2.begin() == address_v4::from_string("192.168.1.1"));
  ASIO_CHECK(addr2.end() != addr2.find(address_v4::from_string("192.168.1.14")));
  ASIO_CHECK(addr2.end() == addr2.find(address_v4::from_string("192.168.1.15")));
  ASIO_CHECK(std::distance(addr3.begin(),addr3.end()) == 14);
  ASIO_CHECK(*addr3.begin() == address_v4::from_string("192.168.1.17"));
  ASIO_CHECK(addr3.end() != addr3.find(address_v4::from_string("192.168.1.30")));
  ASIO_CHECK(addr3.end() == addr3.find(address_v4::from_string("192.168.1.31")));
#endif
}

} // namespace ip_network_v4_runtime

//------------------------------------------------------------------------------

ASIO_TEST_SUITE
(
  "ip/network_v4",
  ASIO_TEST_CASE(ip_network_v4_compile::test)
  ASIO_TEST_CASE(ip_network_v4_runtime::test)
)
