//
// ipv4_host_resolver_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/bind.hpp>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void handle_accept(const error& err)
{
  UNIT_TEST_CHECK(!err);
}

void handle_connect(const error& err)
{
  UNIT_TEST_CHECK(!err);
}

bool test_if_hosts_equivalent(const ipv4::host& h1, const ipv4::host& h2)
{
  // Note: names and aliases may differ between hosts depending on whether the
  // name returned by the OS is fully qualified or not. Therefore we will test
  // hosts for equivalence by checking the list of addresses only.

  if (h1.addresses.size() != h2.addresses.size())
    return false;

  std::vector<ipv4::address> addresses1(h1.addresses);
  std::sort(addresses1.begin(), addresses1.end());

  std::vector<ipv4::address> addresses2(h2.addresses);
  std::sort(addresses2.begin(), addresses2.end());

  for (size_t i = 0; i < addresses1.size(); ++i)
    if (addresses1[i] != addresses2[i])
      return false;

  return true;
}

void ipv4_host_resolver_test()
{
  demuxer d;

  ipv4::host_resolver resolver(d);

  ipv4::host h1;
  resolver.get_local_host(h1);

  ipv4::host h2;
  resolver.get_local_host(h2, throw_error_if(the_error != error::success));

  UNIT_TEST_CHECK(test_if_hosts_equivalent(h1, h2));

  ipv4::host h3;
  resolver.get_host_by_address(h3, h1.addresses[0]);

  ipv4::host h4;
  resolver.get_host_by_address(h4, h1.addresses[0],
      throw_error_if(the_error != error::success));

  UNIT_TEST_CHECK(test_if_hosts_equivalent(h3, h4));
  UNIT_TEST_CHECK(test_if_hosts_equivalent(h1, h3));

  ipv4::host h5;
  resolver.get_host_by_name(h5, h1.name);

  ipv4::host h6;
  resolver.get_host_by_name(h6, h1.name,
      throw_error_if(the_error != error::success));

  UNIT_TEST_CHECK(test_if_hosts_equivalent(h5, h6));
  UNIT_TEST_CHECK(test_if_hosts_equivalent(h1, h5));
}

UNIT_TEST(ipv4_host_resolver_test)
