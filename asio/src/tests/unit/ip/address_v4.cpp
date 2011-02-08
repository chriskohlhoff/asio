//
// address_v4.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/ip/address_v4.hpp"

#include "../unit_test.hpp"
#include <sstream>

//------------------------------------------------------------------------------

// ip_address_v4_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ip::address_v4 compile and link correctly. Runtime failures are ignored.

namespace ip_address_v4_compile {

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  try
  {
    asio::error_code ec;

    // address_v4 constructors.

    ip::address_v4 addr1;
    const ip::address_v4::bytes_type const_bytes_value = { { 127, 0, 0, 1 } };
    ip::address_v4 addr2(const_bytes_value);
    const unsigned long const_ulong_value = 0x7F00001;
    ip::address_v4 addr3(const_ulong_value);

    // address_v4 functions.

    bool b = addr1.is_class_a();
    (void)b;

    b = addr1.is_class_b();
    (void)b;

    b = addr1.is_class_c();
    (void)b;

    b = addr1.is_multicast();
    (void)b;

    ip::address_v4::bytes_type bytes_value = addr1.to_bytes();
    (void)bytes_value;

    unsigned long ulong_value = addr1.to_ulong();
    (void)ulong_value;

    std::string string_value = addr1.to_string();
    string_value = addr1.to_string(ec);

    // address_v4 static functions.

    addr1 = ip::address_v4::from_string("127.0.0.1");
    addr1 = ip::address_v4::from_string("127.0.0.1", ec);
    addr1 = ip::address_v4::from_string(string_value);
    addr1 = ip::address_v4::from_string(string_value, ec);

    addr1 = ip::address_v4::any();

    addr1 = ip::address_v4::loopback();

    addr1 = ip::address_v4::broadcast();

    addr1 = ip::address_v4::broadcast(addr2, addr3);

    addr1 = ip::address_v4::netmask(addr2);

    // address_v4 comparisons.

    b = (addr1 == addr2);
    (void)b;

    b = (addr1 != addr2);
    (void)b;

    b = (addr1 < addr2);
    (void)b;

    b = (addr1 > addr2);
    (void)b;

    b = (addr1 <= addr2);
    (void)b;

    b = (addr1 >= addr2);
    (void)b;

    // address_v4 I/O.

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

} // namespace ip_address_v4_compile

//------------------------------------------------------------------------------

test_suite* init_unit_test_suite(int, char*[])
{
  test_suite* test = BOOST_TEST_SUITE("ip/address_v4");
  test->add(BOOST_TEST_CASE(&ip_address_v4_compile::test));
  return test;
}
