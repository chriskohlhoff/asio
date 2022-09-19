// Copyright (c) 2022 Klemens D. Morgenstern
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/early_complete.hpp"
#include "asio/steady_timer.hpp"

#include "./unit_test.hpp"

struct dummy_init
{
  template<typename Func>
  auto complete_early(Func && func) -> decltype(std::declval<Func>()(int()));

  bool can_complete_early();

};

void trait_test()
{
  using t1 = asio::has_early_completion<dummy_init, void()>;
  using t2 = asio::has_early_completion<dummy_init, void(int)>;
  ASIO_CHECK(!t1::value);
  ASIO_CHECK(t2::value);
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

ASIO_TEST_SUITE
(
    "early_completion",
    ASIO_TEST_CASE(trait_test)
    ASIO_TEST_CASE(timer_test)
)
