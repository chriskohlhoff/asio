//
// bind_immediate_executor.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#include "asio/bind_immediate_executor.hpp"

#include "asio/dispatch.hpp"
#include "asio/io_context.hpp"
#include "unit_test.hpp"

#if defined(ASIO_HAS_BOOST_DATE_TIME)
# include "asio/deadline_timer.hpp"
#else // defined(ASIO_HAS_BOOST_DATE_TIME)
# include "asio/steady_timer.hpp"
#endif // defined(ASIO_HAS_BOOST_DATE_TIME)

#if defined(ASIO_HAS_BOOST_BIND)
# include <boost/bind/bind.hpp>
#else // defined(ASIO_HAS_BOOST_BIND)
# include <functional>
#endif // defined(ASIO_HAS_BOOST_BIND)

using namespace asio;

#if defined(ASIO_HAS_BOOST_BIND)
namespace bindns = boost;
#else // defined(ASIO_HAS_BOOST_BIND)
namespace bindns = std;
#endif

struct initiate_immediate
{
  template <typename Handler>
  void operator()(ASIO_MOVE_ARG(Handler) handler, io_context* ctx) const
  {
    typename associated_immediate_executor<
      Handler, io_context::executor_type>::type ex =
        get_associated_immediate_executor(handler, ctx->get_executor());
    dispatch(ex, ASIO_MOVE_CAST(Handler)(handler));
  }
};

template <ASIO_COMPLETION_TOKEN_FOR(void()) Token>
ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(Token, void())
async_immediate(io_context& ctx, ASIO_MOVE_ARG(Token) token)
  ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX((
    async_initiate<Token, void()>(declval<initiate_immediate>(), token)))
{
  return async_initiate<Token, void()>(initiate_immediate(), token, &ctx);
}

void increment(int* count)
{
  ++(*count);
}

void bind_immediate_executor_to_function_object_test()
{
  io_context ioc1;
  io_context ioc2;

  int count = 0;

  async_immediate(ioc1,
      bind_immediate_executor(
        ioc2.get_executor(),
        bindns::bind(&increment, &count)));

  ioc1.run();

  ASIO_CHECK(count == 0);

  ioc2.run();

  ASIO_CHECK(count == 1);
}

struct incrementer_token_v1
{
  explicit incrementer_token_v1(int* c) : count(c) {}
  int* count;
};

struct incrementer_handler_v1
{
  explicit incrementer_handler_v1(incrementer_token_v1 t) : count(t.count) {}
  void operator()(){ increment(count); }
  int* count;
};

namespace asio {

template <>
class async_result<incrementer_token_v1, void()>
{
public:
  typedef incrementer_handler_v1 completion_handler_type;
  typedef void return_type;
  explicit async_result(completion_handler_type&) {}
  return_type get() {}
};

} // namespace asio

void bind_immediate_executor_to_completion_token_v1_test()
{
  io_context ioc1;
  io_context ioc2;

  int count = 0;

  async_immediate(ioc1,
      bind_immediate_executor(
        ioc2.get_executor(),
        incrementer_token_v1(&count)));

  ioc1.run();

  ASIO_CHECK(count == 0);

  ioc2.run();

  ASIO_CHECK(count == 1);
}

struct incrementer_token_v2
{
  explicit incrementer_token_v2(int* c) : count(c) {}
  int* count;
};

namespace asio {

template <>
class async_result<incrementer_token_v2, void()>
{
public:
#if !defined(ASIO_HAS_RETURN_TYPE_DEDUCTION)
  typedef void return_type;
#endif // !defined(ASIO_HAS_RETURN_TYPE_DEDUCTION)

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename Initiation, typename... Args>
  static void initiate(Initiation initiation,
      incrementer_token_v2 token, ASIO_MOVE_ARG(Args)... args)
  {
    initiation(bindns::bind(&increment, token.count),
        ASIO_MOVE_CAST(Args)(args)...);
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename Initiation>
  static void initiate(Initiation initiation, incrementer_token_v2 token)
  {
    initiation(bindns::bind(&increment, token.count));
  }

#define ASIO_PRIVATE_INITIATE_DEF(n) \
  template <typename Initiation, ASIO_VARIADIC_TPARAMS(n)> \
  static return_type initiate(Initiation initiation, \
      incrementer_token_v2 token, ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    initiation(bindns::bind(&increment, token.count), \
        ASIO_VARIADIC_MOVE_ARGS(n)); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_INITIATE_DEF)
#undef ASIO_PRIVATE_INITIATE_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)
};

} // namespace asio

void bind_immediate_executor_to_completion_token_v2_test()
{
  io_context ioc1;
  io_context ioc2;

  int count = 0;

  async_immediate(ioc1,
      bind_immediate_executor(
        ioc2.get_executor(),
        incrementer_token_v2(&count)));

  ioc1.run();

  ASIO_CHECK(count == 0);

  ioc2.run();

  ASIO_CHECK(count == 1);
}

ASIO_TEST_SUITE
(
  "bind_immediate_executor",
  ASIO_TEST_CASE(bind_immediate_executor_to_function_object_test)
  ASIO_TEST_CASE(bind_immediate_executor_to_completion_token_v1_test)
  ASIO_TEST_CASE(bind_immediate_executor_to_completion_token_v2_test)
)
