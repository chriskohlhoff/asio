//
// address_v6_test.cpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/ip/address_v6.hpp"

#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// ip_address_v6_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ip::address_v6 compile and link correctly. Runtime failures are ignored.

namespace ip_address_v6_compile {

using namespace asio;

void error_handler(const error&)
{
}

void test()
{
  try
  {
    // address_v6 constructors.

    ip::address_v6 addr1;
    const ip::address_v6::bytes_type const_bytes_value = { 0 };
    ip::address_v6 addr2(const_bytes_value);

    // address_v6 functions.

    unsigned long scope_id = addr1.scope_id();
    addr1.scope_id(scope_id);

    bool b = addr1.is_unspecified();

    b = addr1.is_loopback();

    b = addr1.is_multicast();

    b = addr1.is_link_local();

    b = addr1.is_site_local();

    b = addr1.is_v4_mapped();

    b = addr1.is_v4_compatible();

    b = addr1.is_multicast_node_local();

    b = addr1.is_multicast_link_local();

    b = addr1.is_multicast_site_local();

    b = addr1.is_multicast_org_local();

    b = addr1.is_multicast_global();

    ip::address_v6::bytes_type bytes_value = addr1.to_bytes();

    std::string string_value = addr1.to_string();
    string_value = addr1.to_string(error_handler);

    ip::address_v4 addr3 = addr1.to_v4();

    // address_v6 static functions.

    addr1 = ip::address_v6::from_string("0::0");
    addr1 = ip::address_v6::from_string("0::0", error_handler);
    addr1 = ip::address_v6::from_string(string_value);
    addr1 = ip::address_v6::from_string(string_value, error_handler);

    addr1 = ip::address_v6::any();

    addr1 = ip::address_v6::loopback();

    addr1 = ip::address_v6::v4_mapped(addr3);

    addr1 = ip::address_v6::v4_compatible(addr3);

    // address_v6 comparisons.

    b = (addr1 == addr2);

    b = (addr1 != addr2);

    b = (addr1 < addr2);

    b = (addr1 > addr2);

    b = (addr1 <= addr2);

    b = (addr1 >= addr2);

    // address_v6 I/O.

    std::ostringstream os;
    os << addr1;

#if !defined(BOOST_NO_STD_WSTREAMBUF)
    std::wostringstream wos;
    wos << addr1;
#endif // !defined(BOOST_NO_STD_WSTREAMBUF)
  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_address_v6_compile

//------------------------------------------------------------------------------

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("ip/address_v6");
  test->add(BOOST_TEST_CASE(&ip_address_v6_compile::test));
  return test;
}
