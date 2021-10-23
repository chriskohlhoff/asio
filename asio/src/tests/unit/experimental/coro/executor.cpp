//
// experimental/coro/executor.cpp
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
#include "asio/thread_pool.hpp"
#include "asio/experimental/coro.hpp"

#include "asio/io_context.hpp"
#include "../../unit_test.hpp"

using namespace asio::experimental;

namespace coro
{

#define ASIO_CHECKPOINT() ASIO_TEST_IOSTREAM << __FILE__ << "(" << __LINE__ << "): " \
                << asio::detail::test_name() << ": " \
                << "Checkpoint" << std::endl;

template<typename T>
void different_execs()
{
  asio::io_context ctx;
  asio::thread_pool th_ctx{1u};

  auto o = std::make_optional(asio::prefer(th_ctx.get_executor(), asio::execution::outstanding_work.tracked));


  static bool ran_inner = false, ran_outer = false;
  auto c_inner =
        [](asio::any_io_executor e) -> asio::experimental::coro<T>
        {
            auto p = e.target<asio::thread_pool::executor_type>();
            ASIO_CHECKPOINT();
            ASIO_CHECK(p);
            ASIO_CHECK(p->running_in_this_thread());
            ran_inner = true;
            co_return ;
        };

  auto c_outer =
          [](asio::any_io_executor e, auto, asio::experimental::coro<T> tp) -> asio::experimental::coro<void>
          {
            auto p = e.target<asio::io_context::executor_type>();
            ASIO_CHECK(p);
            ASIO_CHECK(p->running_in_this_thread());
            ASIO_CHECKPOINT();
            co_await tp;
            ASIO_CHECKPOINT();
            ASIO_CHECK(p->running_in_this_thread());
            ran_outer = true;
          };

  bool ran = false;
  std::exception_ptr ex;

  auto c = c_outer(ctx.get_executor(), 10, c_inner(th_ctx.get_executor()));
  c.async_resume([&](std::exception_ptr e)
      {
        ASIO_CHECK(!e);
        ASIO_CHECKPOINT();
        ran = true;
      });

  ASIO_CHECK(!ran);
  ctx.run();
  o.reset();
  ASIO_CHECK(ran);
  ASIO_CHECK(ran_inner);
  ASIO_CHECK(ran_outer);
  ASIO_CHECK(!ex);

  th_ctx.stop();
  th_ctx.join();
}

}



ASIO_TEST_SUITE
(
        "coro/partial",
        ASIO_TEST_CASE(::coro::different_execs<void>)
        ASIO_TEST_CASE(::coro::different_execs<int()>)
)