//
// host_resolver_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/ipv4/host_resolver.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <boost/bind.hpp>
#include "asio.hpp"
#include "../unit_test.hpp"

using namespace asio;

void handle_accept(const error& err)
{
  BOOST_CHECK(!err);
}

void handle_connect(const error& err)
{
  BOOST_CHECK(!err);
}

void handle_get_host_by_address(const error& err)
{
  BOOST_CHECK(!err);
}

void handle_get_host_by_name(const error& err)
{
  BOOST_CHECK(!err);
}

bool test_if_hosts_equal(const ipv4::host& h1, const ipv4::host& h2)
{
  if (h1.name() != h2.name())
    return false;

  if (h1.alternate_name_count() != h2.alternate_name_count())
    return false;

  for (size_t i = 0; i < h1.alternate_name_count(); ++i)
    if (h1.alternate_name(i) != h2.alternate_name(i))
      return false;

  if (h1.address_count() != h2.address_count())
    return false;

  for (size_t j = 0; j < h1.address_count(); ++j)
    if (h1.address(j) != h2.address(j))
      return false;

  return true;
}

// This function is used to check whether any of the addresses in the first host
// are found in the second host. This is needed because not all of the network
// interfaces on a system may be known to a DNS server, and not all of the
// addresses associated with a DNS record may belong to a single system.
bool test_if_addresses_intersect(const ipv4::host& h1, const ipv4::host& h2)
{
  std::vector<ipv4::address> addresses1;
  for (size_t i = 0; i < h1.address_count(); ++i)
    addresses1.push_back(h1.address(i));
  std::sort(addresses1.begin(), addresses1.end());

  std::vector<ipv4::address> addresses2;
  for (size_t j = 0; j < h2.address_count(); ++j)
    addresses2.push_back(h2.address(j));
  std::sort(addresses2.begin(), addresses2.end());

  std::vector<ipv4::address> intersection;
  set_intersection(addresses1.begin(), addresses1.end(),
      addresses2.begin(), addresses2.end(), std::back_inserter(intersection));

  return !intersection.empty();
}

void ipv4_host_resolver_test()
{
  demuxer d;

  ipv4::host_resolver resolver(d);

  ipv4::host h1;
  resolver.get_local_host(h1);

  ipv4::host h2;
  resolver.get_local_host(h2, throw_error());

  BOOST_CHECK(test_if_hosts_equal(h1, h2));

  ipv4::host h3;
  resolver.get_host_by_address(h3, h1.address(0));

  ipv4::host h4;
  resolver.get_host_by_address(h4, h1.address(0), throw_error());

  BOOST_CHECK(test_if_hosts_equal(h3, h4));
  BOOST_CHECK(test_if_addresses_intersect(h1, h3));

  ipv4::host h5;
  resolver.get_host_by_name(h5, h1.name());

  ipv4::host h6;
  resolver.get_host_by_name(h6, h1.name(), throw_error());

  BOOST_CHECK(test_if_hosts_equal(h5, h6));
  BOOST_CHECK(test_if_addresses_intersect(h1, h5));

  ipv4::host h7;
  resolver.async_get_host_by_address(h7, h1.address(0),
      handle_get_host_by_address);
  d.reset();
  d.run();

  BOOST_CHECK(test_if_hosts_equal(h3, h7));
  BOOST_CHECK(test_if_addresses_intersect(h1, h7));

  ipv4::host h8;
  resolver.async_get_host_by_name(h8, h1.name(), handle_get_host_by_name);
  d.reset();
  d.run();

  BOOST_CHECK(test_if_hosts_equal(h5, h8));
  BOOST_CHECK(test_if_addresses_intersect(h1, h8));
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("ipv4/host_resolver");
  test->add(BOOST_TEST_CASE(&ipv4_host_resolver_test));
  return test;
}
