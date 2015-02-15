//
// unit_test.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef UNIT_TEST_HPP
#define UNIT_TEST_HPP

#include "asio/detail/config.hpp"

#if defined(__sun)
# include <stdlib.h> // Needed for lrand48.
#endif // defined(__sun)

#if defined(__BORLANDC__)

// Prevent use of intrinsic for strcmp.
# include <cstring>
# undef strcmp
 
// Suppress error about condition always being true.
# pragma option -w-ccc

#endif // defined(__BORLANDC__)

#if defined(ASIO_MSVC)
# pragma warning (push)
# pragma warning (disable:4244)
# pragma warning (disable:4702)
#endif // defined(ASIO_MSVC)

#if defined(ASIO_STANDALONE)

#include <cassert>
#include <iostream>

#if defined(NDEBUG)
# error NDEBUG must not be defined when building these unit tests
#endif // defined(NDEBUG)

#define ASIO_CHECK(expr) assert(expr)

#define ASIO_CHECK_MESSAGE(expr, msg) \
  do { if (!(expr)) { std::cout << msg << std::endl; assert(expr); } } while (0)

#define ASIO_WARN_MESSAGE(expr, msg) \
  do { if (!(expr)) { std::cout << msg << std::endl; } } while (0)

#define ASIO_ERROR(msg) assert(0 && msg)

#define ASIO_TEST_SUITE(name, tests) \
  int main() \
  { \
    std::cout << name << " test suite begins" << std::endl; \
    tests \
    std::cout << name << " test suite ends" << std::endl; \
    return 0; \
  }

#define ASIO_TEST_CASE(test) \
  test(); \
  std::cout << #test << " passed" << std::endl;

#define ASIO_COMPILE_TEST_CASE(test) \
  compile_test<&test>(); \
  std::cout << #test << " passed" << std::endl;

#else // defined(ASIO_STANDALONE)

#include <boost/test/unit_test.hpp>
using boost::unit_test::test_suite;

#define ASIO_CHECK(expr) BOOST_CHECK(expr)

#define ASIO_CHECK_MESSAGE(expr, msg) BOOST_CHECK_MESSAGE(expr, msg)

#define ASIO_WARN_MESSAGE(expr, msg) BOOST_WARN_MESSAGE(expr, msg)

#define ASIO_ERROR(expr) BOOST_ERROR(expr)

#define ASIO_TEST_SUITE(name, tests) \
  test_suite* init_unit_test_suite(int, char*[]) \
  { \
    test_suite* t = BOOST_TEST_SUITE(name); \
    tests \
    return t; \
  }

#define ASIO_TEST_CASE(test) \
  t->add(BOOST_TEST_CASE(&test));

#define ASIO_COMPILE_TEST_CASE(test) \
  t->add(BOOST_TEST_CASE(&compile_test<&test>));

#endif // defined(ASIO_STANDALONE)

#if defined(ASIO_MSVC)
# pragma warning (pop)
#endif // defined(ASIO_MSVC)

inline void null_test()
{
}

template <void (*)()>
inline void compile_test()
{
}

#if defined(__GNUC__) && defined(_AIX)

// AIX needs this symbol defined in asio, even if it doesn't do anything.
int test_main(int, char**)
{
}

#endif // defined(__GNUC__) && defined(_AIX)

#endif // UNIT_TEST_HPP
