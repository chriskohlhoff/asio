//
// piping.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/piping.hpp"

#include "asio/deferred.hpp"
#include "asio/eager.hpp"
#include "asio/error_code.hpp"
#include "asio/io_context.hpp"
#include "asio/steady_timer.hpp"
#include <chrono>
#include "unit_test.hpp"

namespace piping_test {
struct not_pipeable
{
};

struct user_op
{
  int* result;

  template <typename Token>
  int operator()(Token&& token) const
  {
    *result = token(7);
    return *result;
  }
};

struct user_token
{
  int operator()(int value) const { return value * 2; }
};
} // namespace piping_test

namespace asio {
template <>
struct supports_piping<piping_test::user_op> : true_type
{
};
} // namespace asio

namespace piping_test {
void piping_trait()
{
  using namespace asio;

  io_context ctx;
  steady_timer timer(ctx);

  typedef decltype(timer.async_wait(deferred)) deferred_op;
  ASIO_CHECK(supports_piping<deferred_op>::value);
  ASIO_CHECK(supports_piping<deferred_values<int> >::value);

  ASIO_CHECK(!supports_piping<deferred_t>::value);

  typedef decltype(timer.async_wait(eager)) eager_op;
  ASIO_CHECK(supports_piping<eager_op>::value);

  ASIO_CHECK(!is_deferred<eager_op>::value);

  ASIO_CHECK(!supports_piping<not_pipeable>::value);
  ASIO_CHECK(!supports_piping<int>::value);
  ASIO_CHECK(!supports_piping<steady_timer>::value);

  ASIO_CHECK(supports_piping<user_op>::value);

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)
  ASIO_CHECK(supports_piping_v<deferred_op>);
  ASIO_CHECK(supports_piping_v<eager_op>);
  ASIO_CHECK(!supports_piping_v<not_pipeable>);
#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)
}

void piping_operator()
{
  using namespace asio;

  int result = 0;
  int returned = user_op{&result} | user_token();
  ASIO_CHECK(result == 14);
  ASIO_CHECK(returned == 14);
}

void piping_deferred_chain()
{
  using namespace asio;

  io_context ctx;
  steady_timer timer(ctx);
  timer.expires_after(std::chrono::milliseconds(0));

  int got = -1;
  timer.async_wait(deferred)
    | deferred([](asio::error_code){ return deferred.values(42); })
    | [&](int value){ got = value; };

  ctx.run();

  ASIO_CHECK(got == 42);
}

void piping_eager_through_deferred()
{
  using namespace asio;

  io_context ctx;
  steady_timer timer(ctx);
  timer.expires_after(std::chrono::milliseconds(0));

  int got = -1;
  timer.async_wait(eager)
    | deferred([](asio::error_code){ return deferred.values(99); })
    | [&](int value){ got = value; };

  ctx.run();

  ASIO_CHECK(got == 99);
}
} // namespace piping_test

ASIO_TEST_SUITE
(
  "piping",
  ASIO_TEST_CASE(piping_test::piping_trait)
  ASIO_TEST_CASE(piping_test::piping_operator)
  ASIO_TEST_CASE(piping_test::piping_deferred_chain)
  ASIO_TEST_CASE(piping_test::piping_eager_through_deferred)
)
