//
// basic_resolver_results.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/ip/basic_resolver_results.hpp"

#include <cstring>
#include "asio/ip/tcp.hpp"
#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// ip_basic_resolver_results_runtime test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that results created from an addrinfo list with
// no usable entries are empty and can be iterated.

namespace ip_basic_resolver_results_runtime {

void test()
{
  using namespace std; // For memset.
  using asio::ip::tcp;

  // An IPv4 entry whose address is too short to be used.
  asio::detail::addrinfo_type info;
  memset(&info, 0, sizeof(info));
  info.ai_family = ASIO_OS_DEF(AF_INET);
  info.ai_addrlen = 0;

  tcp::resolver::results_type results =
    tcp::resolver::results_type::create(&info, "host", "service");

  ASIO_CHECK(results.size() == 0);
  ASIO_CHECK(results.empty());
  ASIO_CHECK(results.begin() == results.end());
  ASIO_CHECK(results.cbegin() == results.cend());
  ASIO_CHECK(results == tcp::resolver::results_type());

  std::size_t count = 0;
  for (tcp::resolver::results_type::const_iterator i = results.begin();
      i != results.end() && count < 2; ++i)
    ++count;
  ASIO_CHECK(count == 0);
}

} // namespace ip_basic_resolver_results_runtime

//------------------------------------------------------------------------------

ASIO_TEST_SUITE
(
  "ip/basic_resolver_results",
  ASIO_TEST_CASE(ip_basic_resolver_results_runtime::test)
)
