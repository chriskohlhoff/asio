//
// cidr_address_v4.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2012 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/ip/cidr_address_v4.hpp"

#include "../unit_test.hpp"
#include <sstream>

//------------------------------------------------------------------------------

// ip_cidr_address_v4_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ip::cidr_address_v4 compile and link correctly. Runtime failures are ignored.

namespace ip_cidr_address_v4_compile {

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  try
  {
    asio::error_code ec;

    // cidr_address_v4 constructors.

    ip::cidr_address_v4 range1( ip::address_v4::from_string("192.168.1.0"),
                                ip::address_v4::from_string("255.255.255.0") );

    ip::cidr_address_v4 range2( ip::address_v4::from_string("192.168.1.0"), 32);

    ip::address_v4 addr;
    ip::address_iterator_v4 i( addr);

    addr = range2.network();
    (void) addr;

    addr = range2.netmask();
    (void) addr;

    addr = range2.broadcast();
    (void) addr;

    i = range2.begin();
    (void) i;

    i = range2.end();
    (void) i;

    i = range2.find( ip::address_v4::from_string("192.168.1.250") );
    (void) i;

  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_cidr_address_v4_compile

//------------------------------------------------------------------------------

// ip_cidr_address_v4_runtime test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that the various public member functions meet the
// necessary postconditions.

namespace ip_cidr_address_v4_runtime {

void test()
{
  using asio::ip::address_v4;
  using asio::ip::address_iterator_v4;
  using asio::ip::cidr_address_v4;

  // calculate prefix length
  ASIO_CHECK( cidr_address_v4::calculate_prefix_length( address_v4::from_string("255.255.255.0") ) == 24);
  ASIO_CHECK( cidr_address_v4::calculate_prefix_length( address_v4::from_string("255.255.255.192") ) == 26);
  ASIO_CHECK( cidr_address_v4::calculate_prefix_length( address_v4::from_string("128.0.0.0") ) == 1);

  std::string msg;
  try
  { cidr_address_v4::calculate_prefix_length( address_v4::from_string("255.255.255.1") ); }
  catch( std::out_of_range const& ex)
  { msg = ex.what(); }
  ASIO_CHECK( msg == std::string("prefix from netmask") );

  msg.clear();
  try
  { cidr_address_v4::calculate_prefix_length( address_v4::from_string("0.255.255.0") ); }
  catch( std::out_of_range const& ex)
  { msg = ex.what(); }
  ASIO_CHECK( msg == std::string("prefix from netmask") );


  // calculate netmask
  ASIO_CHECK(cidr_address_v4::calculate_netmask(23) == address_v4::from_string("255.255.254.0"));
  ASIO_CHECK(cidr_address_v4::calculate_netmask(12) == address_v4::from_string("255.240.0.0"));
  ASIO_CHECK(cidr_address_v4::calculate_netmask(24) == address_v4::from_string("255.255.255.0"));
  ASIO_CHECK(cidr_address_v4::calculate_netmask(16) == address_v4::from_string("255.255.0.0"));
  ASIO_CHECK(cidr_address_v4::calculate_netmask(8) == address_v4::from_string("255.0.0.0"));
  ASIO_CHECK(cidr_address_v4::calculate_netmask(32) == address_v4::from_string("255.255.255.255"));
  ASIO_CHECK(cidr_address_v4::calculate_netmask(1) == address_v4::from_string("128.0.0.0"));
  ASIO_CHECK(cidr_address_v4::calculate_netmask(0) == address_v4::from_string("0.0.0.0"));

  msg.clear();
  try
  { cidr_address_v4::calculate_netmask(33); }
  catch( std::out_of_range const& ex)
  { msg = ex.what(); }
  ASIO_CHECK( msg == std::string("netmask from prefix") );

  // construct address range from address and prefix length
  ASIO_CHECK(cidr_address_v4(address_v4::from_string("192.168.77.100"), 32).network() == address_v4::from_string("192.168.77.100"));
  ASIO_CHECK(cidr_address_v4(address_v4::from_string("192.168.77.100"), 24).network() == address_v4::from_string("192.168.77.0"));
  ASIO_CHECK(cidr_address_v4(address_v4::from_string("192.168.77.128"), 25).network() == address_v4::from_string("192.168.77.128"));

  // construct address range from string in CIDR notation
  ASIO_CHECK(cidr_address_v4::from_string("192.168.77.100/32").network() == address_v4::from_string("192.168.77.100"));
  ASIO_CHECK(cidr_address_v4::from_string("192.168.77.100/24").network() == address_v4::from_string("192.168.77.0"));
  ASIO_CHECK(cidr_address_v4::from_string("192.168.77.128/25").network() == address_v4::from_string("192.168.77.128"));

  // prefix length
  ASIO_CHECK(cidr_address_v4::from_string("193.99.144.80/24").prefix_length() == 24);
  ASIO_CHECK(cidr_address_v4(address_v4::from_string("193.99.144.80"), 24).prefix_length() == 24);
  ASIO_CHECK(cidr_address_v4(address_v4::from_string("192.168.77.0"), address_v4::from_string("255.255.255.0")).prefix_length() == 24);

  // to string
  std::string a("192.168.77.0/32");
  ASIO_CHECK(cidr_address_v4::from_string(a.c_str()).to_string() == a);
  ASIO_CHECK(cidr_address_v4(address_v4::from_string("192.168.77.10"), 24).to_string() == std::string("192.168.77.10/24"));

  // return host part
  ASIO_CHECK(cidr_address_v4::from_string("192.168.77.11/24").host() == address_v4::from_string("192.168.77.11"));

  // return host in CIDR notation
  ASIO_CHECK(cidr_address_v4::from_string("192.168.78.30/20").host_cidr().to_string() == "192.168.78.30/32");

  // return network in CIDR notation
  ASIO_CHECK(cidr_address_v4::from_string("192.168.78.30/20").network_cidr().to_string() == "192.168.64.0/20");

  // is host
  ASIO_CHECK(cidr_address_v4::from_string("192.168.77.0/32").is_host());
  ASIO_CHECK(!cidr_address_v4::from_string("192.168.77.0/31").is_host());

  // is real subnet of
  ASIO_CHECK(cidr_address_v4::from_string("192.168.0.192/24").is_subnet_of(cidr_address_v4::from_string("192.168.0.0/16")));
  ASIO_CHECK(cidr_address_v4::from_string("192.168.0.0/24").is_subnet_of(cidr_address_v4::from_string("192.168.192.168/16")));
  ASIO_CHECK(cidr_address_v4::from_string("192.168.0.192/24").is_subnet_of(cidr_address_v4::from_string("192.168.192.168/16")));
  ASIO_CHECK(cidr_address_v4::from_string("192.168.0.0/24").is_subnet_of(cidr_address_v4::from_string("192.168.0.0/16")));
  ASIO_CHECK(cidr_address_v4::from_string("192.168.0.0/24").is_subnet_of(cidr_address_v4::from_string("192.168.0.0/23")));
  ASIO_CHECK(cidr_address_v4::from_string("192.168.0.0/24").is_subnet_of(cidr_address_v4::from_string("192.168.0.0/0")));
  ASIO_CHECK(cidr_address_v4::from_string("192.168.0.0/32").is_subnet_of(cidr_address_v4::from_string("192.168.0.0/24")));

  ASIO_CHECK(!cidr_address_v4::from_string("192.168.0.0/32").is_subnet_of(cidr_address_v4::from_string("192.168.0.0/32")));
  ASIO_CHECK(!cidr_address_v4::from_string("192.168.0.0/24").is_subnet_of(cidr_address_v4::from_string("192.168.1.0/24")));
  ASIO_CHECK(!cidr_address_v4::from_string("192.168.0.0/16").is_subnet_of(cidr_address_v4::from_string("192.168.1.0/24")));

  cidr_address_v4 r(cidr_address_v4::from_string("192.168.0.0/24"));
  ASIO_CHECK(!r.is_subnet_of(r));

  cidr_address_v4 addr(cidr_address_v4::from_string("192.168.0.2/24"));
  // network
  ASIO_CHECK(addr.network() == address_v4::from_string("192.168.0.0"));
  // netamsk
  ASIO_CHECK(addr.netmask() == address_v4::from_string("255.255.255.0"));
  // broadcast
  ASIO_CHECK(addr.broadcast() == address_v4::from_string("192.168.0.255"));
  // iterator
  ASIO_CHECK(std::distance(addr.begin(),addr.end()) == 254);
  ASIO_CHECK(*addr.begin() == address_v4::from_string("192.168.0.1"));
  ASIO_CHECK(addr.end() != addr.find(address_v4::from_string("192.168.0.10")));
  ASIO_CHECK(addr.end() == addr.find(address_v4::from_string("192.168.1.10")));
}

} // namespace ip_cidr_address_v4_runtime

//------------------------------------------------------------------------------

ASIO_TEST_SUITE
(
  "ip/cidr_address_v4",
  ASIO_TEST_CASE(ip_cidr_address_v4_compile::test)
  ASIO_TEST_CASE(ip_cidr_address_v4_runtime::test)
)
