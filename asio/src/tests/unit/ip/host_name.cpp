//
// host_name.cpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/ip/host_name.hpp"

#include "asio.hpp"
#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// ip_host_name_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all host_name functions compile and link
// correctly. Runtime failures are ignored.

namespace ip_host_name_compile {

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  try
  {
    asio::error_code ec;

    std::string host_name = ip::host_name();
    std::string host_name2 = ip::host_name(ec);
  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_host_name_compile

//------------------------------------------------------------------------------

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("ip/host_name");
  test->add(BOOST_TEST_CASE(&ip_host_name_compile::test));
  return test;
}
