// Copyright (c) 2022 Klemens D. Morgenstern
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/early_complete.hpp"
#include "asio/co_spawn.hpp"
#include "asio/steady_timer.hpp"
#include "asio/experimental/channel.hpp"
#include "asio/use_awaitable.hpp"

#include "./unit_test.hpp"

struct dummy_init
{
  template<typename Handler>
  bool try_complete(Handler && handler, decltype(std::declval<Handler>()(int())) * = nullptr);
};

struct dummy_init_s
{
  template<typename Handler>
  bool try_complete(Handler && handler, int, decltype(std::declval<Handler>()(int())) * = nullptr);
};

void trait_test()
{
  using t1 = asio::has_early_completion<dummy_init, std::tuple<>, void()>;
  using t2 = asio::has_early_completion<dummy_init, std::tuple<>, void(int)>;
  ASIO_CHECK(!t1::value);
  ASIO_CHECK(t2::value);

  using u1 = asio::has_early_completion<dummy_init_s, std::tuple<int>, void()>;
  using u2 = asio::has_early_completion<dummy_init_s, std::tuple<int>, void(int)>;
  using u3 = asio::has_early_completion<dummy_init_s, std::tuple<>, void(int)>;
  ASIO_CHECK(!u1::value);
  ASIO_CHECK(u2::value);
  ASIO_CHECK(!u3::value);

}

void timer_test()
{
  asio::io_context ctx;
  asio::steady_timer tim{ctx};

  // this thing doesn't dispatch, we're just completing immediately

  bool done = false;
  auto cpl =  [&](asio::error_code ec)
              {
                ASIO_CHECK(!ec);
                done = true;
              };

  tim.async_wait(asio::allow_recursion(cpl));
  ASIO_CHECK(done);
  done = false;

  tim.expires_after(std::chrono::milliseconds(50));
  tim.async_wait(asio::allow_recursion(cpl));
  ASIO_CHECK(!done);
  ctx.run();
  ASIO_CHECK(done);
}


void channel_test()
{
  asio::io_context ctx;
  asio::experimental::channel<void(asio::error_code, int)> chn{ctx, 1};

  // this thing doesn't dispatch, we're just completing immediately
  bool done = false;
  auto cpl =  [&](asio::error_code ec)
  {
    ASIO_CHECK(!ec);
    done = true;
  };

  int reced = 0;
  auto rec =  [&](asio::error_code ec, int i)
  {
    ASIO_CHECK(!ec);
    reced = i;
  };

  chn.async_send(asio::error_code{}, 42, asio::allow_recursion(cpl));
  ASIO_CHECK(done);
  done = false;

  chn.async_receive(asio::allow_recursion(rec));
  ASIO_CHECK(reced == 42);

  chn.async_send(asio::error_code{}, 43, asio::allow_recursion(cpl));
  ASIO_CHECK(done);
  done = false;
  chn.async_send(asio::error_code{}, 44, asio::allow_recursion(cpl));
  ASIO_CHECK(!done);
  chn.async_receive(rec);
  ctx.run();
  ASIO_CHECK(done);
}

asio::awaitable<void> awaitable_test_impl()
{
  asio::steady_timer tim{co_await asio::this_coro::executor};

  bool posted = false;
  asio::post(co_await asio::this_coro::executor, [&]{posted = true;});

  ASIO_CHECK(!posted);
  co_await tim.async_wait(asio::use_awaitable);
  ASIO_CHECK(!posted);

  co_await asio::post(asio::use_awaitable);
  ASIO_CHECK(posted);

}

void awaitable_test()
{
  asio::io_context ctx;
  asio::co_spawn(ctx, awaitable_test_impl(), [](std::exception_ptr e){ASIO_CHECK(!e);});
  ctx.run();
}

ASIO_TEST_SUITE
(
    "early_completion",
    ASIO_TEST_CASE(trait_test)
    ASIO_TEST_CASE(timer_test)
    ASIO_TEST_CASE(channel_test)
    ASIO_TEST_CASE(awaitable_test)
)
