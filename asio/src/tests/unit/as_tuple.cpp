//
// as_tuple.cpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/as_tuple.hpp"

#include "asio/bind_executor.hpp"
#include "asio/io_context.hpp"
#include "asio/post.hpp"
#include "asio/system_timer.hpp"
#include "unit_test.hpp"

void as_tuple_test()
{
#if defined(ASIO_HAS_STD_TUPLE) \
  && defined(ASIO_HAS_VARIADIC_TEMPLATES)
  asio::io_context io1;
  asio::io_context io2;
  asio::system_timer timer1(io1);
  int count = 0;

  timer1.expires_after(asio::chrono::seconds(0));
  timer1.async_wait(
      asio::as_tuple(
        asio::bind_executor(io2.get_executor(),
          [&count](std::tuple<asio::error_code>)
          {
            ++count;
          })));

  ASIO_CHECK(count == 0);

  io1.run();

  ASIO_CHECK(count == 0);

  io2.run();

  ASIO_CHECK(count == 1);
#endif // defined(ASIO_HAS_STD_TUPLE)
       //   && defined(ASIO_HAS_VARIADIC_TEMPLATES)
}

ASIO_TEST_SUITE
(
  "as_tuple",
  ASIO_TEST_CASE(as_tuple_test)
)
