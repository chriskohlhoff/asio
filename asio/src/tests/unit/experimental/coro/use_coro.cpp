//
// experimental/coro/use_coro.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern
//                    (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/experimental/use_coro.hpp"

#include "asio/steady_timer.hpp"
#include <iostream>
#include "../../unit_test.hpp"

using namespace asio::experimental;

namespace coro {

asio::experimental::coro<void() noexcept, int>
poster(asio::any_io_executor exec)
{
  co_await asio::post(exec, use_coro);
  co_return 42;
}


asio::experimental::coro<void() , int>
throwing(asio::any_io_executor exec)
{
  auto res = co_await poster(exec);
  throw std::logic_error("test-exception");
  co_return res;
}

asio::experimental::coro<void(), int>
erroring(asio::any_io_executor exec)
{
  auto t = throwing(exec);
  co_await t.async_resume(use_coro);
  co_return 42;
}

asio::experimental::coro<void(), int>
awaiter(asio::any_io_executor exec)
{
  asio::steady_timer timer{exec};
  co_await timer.async_wait(use_coro);
  co_return 42;
}

asio::experimental::coro<void() noexcept, int>
awaiter_noexcept(asio::any_io_executor exec)
{
  asio::steady_timer timer{exec};
  auto ec = co_await timer.async_wait(use_coro);
  ASIO_CHECK(ec == asio::error_code{});
  co_return 42;
}

void use_coro_direct()
{
  int done = 0;
  asio::io_context ctx;

  auto k = awaiter(ctx.get_executor());
  auto k2 = awaiter_noexcept(ctx.get_executor());
  auto k3 = poster(ctx.get_executor());
  auto k4 = erroring(ctx.get_executor());

  k.async_resume(
      [&](std::exception_ptr ex, int res)
      {
        ASIO_CHECK(!ex);
        ASIO_CHECK(res == 42);
        done++;
      });

  k2.async_resume([&](int res)
       {
         ASIO_CHECK(res == 42);
         done++;
       });

  k3.async_resume([&](int res)
                 {
                   ASIO_CHECK(res == 42);
                   done ++;
                 });

  k4.async_resume([&](std::exception_ptr ex, int )
                  {
                    ASIO_CHECK(ex != nullptr);
                    done ++;
                  });

  ctx.run();
  ASIO_CHECK(done == 4);
}

asio::experimental::coro<void, void>
  use_coro_nested_impl(asio::any_io_executor exec)
{
  ASIO_CHECK(42 == co_await awaiter(exec));
  ASIO_CHECK(42 == co_await awaiter_noexcept(exec));
  ASIO_CHECK(42 == co_await poster(exec));
  bool caught = false;
  try
  {
    co_await erroring(exec);
  }
  catch (std::logic_error & )
  {
    caught = true;
  }
  ASIO_CHECK(caught);
}

void use_coro_nested()
{
  int done = 0;
  asio::io_context ctx;

  auto k = use_coro_nested_impl(ctx.get_executor());
  k.async_resume(
          [&](std::exception_ptr ex)
          {
            ASIO_CHECK(!ex);
            done++;
          });
  ctx.run();
  ASIO_CHECK(done== 1);
}

} // namespace coro

ASIO_TEST_SUITE
(
  "coro/use_coro",
  ASIO_TEST_CASE(::coro::use_coro_direct)
  ASIO_TEST_CASE(::coro::use_coro_nested)
)
