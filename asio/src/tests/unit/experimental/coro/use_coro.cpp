//
// experimental/coro/use_coro.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021-2022 Klemens D. Morgenstern
//                         (klemens dot morgenstern at gmx dot net)
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

#include "asio/detached.hpp"
#include "asio/redirect_error.hpp"
#include "asio/steady_timer.hpp"

#include <iostream>
#include <asio/this_coro.hpp>

#include "../../unit_test.hpp"

using namespace asio::experimental;

namespace coro {

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

void stack_test2()
{
  bool done = false;
  asio::io_context ctx;

  auto k = awaiter(ctx.get_executor());
  auto k2 = awaiter_noexcept(ctx.get_executor());

  k.async_resume(
      [&](std::exception_ptr ex, int res)
      {
        ASIO_CHECK(!ex);
        ASIO_CHECK(res == 42);
        done = true;
      });

  k2.async_resume([&](int res)
       {
         ASIO_CHECK(res == 42);
         done = true;
       });

  ctx.run();
  ASIO_CHECK(done);
}

asio::experimental::coro<void() noexcept, void>
cancel_inner(asio::any_io_executor exec, int &cancel_cnt)
{
  asio::steady_timer st{exec, std::chrono::steady_clock::time_point::max()};

  (co_await asio::this_coro::cancellation_state).slot().assign(
            [&](auto c)
          {
            cancel_cnt ++;
            st.cancel();
            return c;
          });
  asio::error_code ec;

  co_await st.async_wait(
          asio::bind_cancellation_slot(asio::cancellation_slot{},
          asio::redirect_error(asio::experimental::use_coro,ec)));

};

asio::experimental::coro<void() noexcept, void>
cancel_test(asio::any_io_executor exec, int & cancelled)
{
  auto inner = cancel_inner(exec, cancelled);

  co_await inner.async_resume(asio::experimental::use_coro);
}

void cancel_test()
{
  int cancelled = 0;
  asio::io_context ctx;
  auto ct = cancel_test(ctx.get_executor(), cancelled);

  asio::cancellation_signal cs;
  ct.async_resume(asio::bind_cancellation_slot(cs.slot(), asio::detached));

  asio::post(ctx, [&]{cs.emit(asio::cancellation_type::all);});
  ctx.run();

  ASIO_CHECK(cancelled == 1);
}

} // namespace coro

ASIO_TEST_SUITE
(
  "coro/use_coro",
  ASIO_TEST_CASE(::coro::stack_test2)
  ASIO_TEST_CASE(::coro::cancel_test)
)
