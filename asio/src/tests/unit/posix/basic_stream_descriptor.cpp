//
// basic_stream_descriptor.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#include "asio/posix/basic_stream_descriptor.hpp"

#include "asio.hpp"
#include "../unit_test.hpp"

test_suite* init_unit_test_suite(int, char*[])
{
  test_suite* test = BOOST_TEST_SUITE("posix/basic_stream_descriptor");
  test->add(BOOST_TEST_CASE(&null_test));
  return test;
}
