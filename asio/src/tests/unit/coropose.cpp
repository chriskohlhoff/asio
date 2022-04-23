// Copyright (c) 2021 Klemens D. Morgenstern
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include <asio/this_coro.hpp>
#include <asio/post.hpp>
#include <asio/io_context.hpp>
#include "asio/coropose.hpp"

#include "unit_test.hpp"


template<typename ContextOrExecutor, typename CompletionToken>
auto async_divide(ContextOrExecutor && ctx, int x, int y,
               CompletionToken && completion_token,
               asio::async_coropose_tag<void(int)> = {})
  -> asio::async_coropose_t<void(int), CompletionToken>
{
  co_await asio::post(ctx, co_await asio::this_coro::token);

  if (y == 0)
    throw std::runtime_error("divide by zero");

  co_return x / y;
}

template<typename ContextOrExecutor, typename CompletionToken>
auto async_throw(ContextOrExecutor && ctx, int x,
               CompletionToken && completion_token,
               asio::async_coropose_tag<void(int)> = {})
-> asio::async_coropose_t<void(asio::error_code, int), CompletionToken>
{
  co_await asio::post(ctx, co_await asio::this_coro::token);

  co_return std::tuple{asio::error_code{asio::error::access_denied}, x};
}


void coropose()
{
  asio::io_context ctx;

  int res = 0;
  async_divide(ctx, 144, 12, [&](int i){ res = i;});

  asio::co_spawn(ctx, async_divide(ctx, 100, 10, asio::use_awaitable), [](std::exception_ptr e, int i) {ASIO_CHECK(i == 10);});

  auto cr = async_divide(ctx, 75, 5, asio::experimental::use_coro);
  cr.async_resume([]( int i) {ASIO_CHECK(i == 15);});

  ctx.run();
  ASIO_CHECK(res == 12);

  try
  {
    ctx.restart();
    async_divide(ctx, 1, 0, [&](int i){ res = i;});
    ctx.run();
    ASIO_CHECK(false);

  }
  catch(std::runtime_error & re)
  {
    ASIO_CHECK(re.what() == std::string("divide by zero"));
  }

}




ASIO_TEST_SUITE
(
  "corpose",
  ASIO_TEST_CASE(coropose)
)
