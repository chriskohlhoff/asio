//
// as_tuple.cpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2022 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/error_as_deferred.hpp"

#include "asio/awaitable.hpp"
#include "asio/io_context.hpp"
#include "asio/system_timer.hpp"
#include "unit_test.hpp"

void error_as_deferred_test_no_error()
{
#if defined(ASIO_HAS_VARIADIC_TEMPLATES)
  asio::io_context ioc;
  asio::system_timer timer(ioc);
  asio::error_code ec;
  int count = 0;
  timer.expires_after(asio::chrono::seconds(0));
  timer.async_wait(ec)([&count]{count ++;});
  ASIO_CHECK(count == 0);
  ASIO_CHECK(!ec);

  ioc.run();

  ASIO_CHECK(!ec);
  ASIO_CHECK(count == 1);
#endif // defined(ASIO_HAS_STD_TUPLE)
       //   && defined(ASIO_HAS_VARIADIC_TEMPLATES)
}

void error_as_deferred_test_error()
{
#if defined(ASIO_HAS_VARIADIC_TEMPLATES)
  asio::io_context ioc;
  asio::system_timer timer(ioc);
  asio::error_code ec;
  int count = 0;
  timer.expires_after(asio::chrono::seconds(1000));
  timer.async_wait(ec)([&count]{count ++;});
  timer.cancel();
  ASIO_CHECK(count == 0);
  ASIO_CHECK(!ec);

  ioc.run();

  ASIO_CHECK(ec == asio::error::operation_aborted);
  ASIO_CHECK(count == 1);
#endif // defined(ASIO_HAS_STD_TUPLE)
  //   && defined(ASIO_HAS_VARIADIC_TEMPLATES)
}

ASIO_TEST_SUITE
(
  "error_as_deferred",
  ASIO_TEST_CASE(error_as_deferred_test_no_error)
  ASIO_TEST_CASE(error_as_deferred_test_error)
)
