//
// experimental/awaitable_specific_ptr.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2024 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Prevent link dependency on the Boost.System library.
#if !defined(BOOST_SYSTEM_NO_DEPRECATED)
#define BOOST_SYSTEM_NO_DEPRECATED
#endif // !defined(BOOST_SYSTEM_NO_DEPRECATED)

// Test that header file is self-contained.
#include "asio/experimental/awaitable_specific_ptr.hpp"
#include "asio/awaitable.hpp"
#include "asio/co_spawn.hpp"
#include "asio/detached.hpp"
#include "asio/io_context.hpp"
#include "asio/post.hpp"
#include "../unit_test.hpp"

asio::experimental::awaitable_specific_ptr<int> test_ptr;

asio::awaitable<void> do_awaitable_specific_ptr(int val)
{
  int *stored = co_await test_ptr.get();

  ASIO_CHECK(stored == nullptr);
  co_await test_ptr.reset(new int(val));
  // co_await std::suspend_always;

  stored = co_await test_ptr.get();
  ASIO_CHECK(stored != nullptr);
  ASIO_CHECK(*stored == val);
  // co_await std::suspend_always;

  stored = co_await test_ptr.release();
  ASIO_CHECK(stored != nullptr);
  ASIO_CHECK(*stored == val);
  delete stored;
  // co_await std::suspend_always;

  stored = co_await test_ptr.get();
  ASIO_CHECK(stored == nullptr);
}

void test_awaitable_specific_ptr()
{
  asio::io_context ctx;
  co_spawn(ctx, do_awaitable_specific_ptr(1), asio::detached);
  co_spawn(ctx, do_awaitable_specific_ptr(2), asio::detached);
  ctx.run();
}

ASIO_TEST_SUITE
(
    "experimental/awaitable_specific_ptr",
    ASIO_TEST_CASE(test_awaitable_specific_ptr)
)