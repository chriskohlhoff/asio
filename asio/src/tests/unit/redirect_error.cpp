//
// redirect_error.cpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/redirect_error.hpp"

#include "unit_test.hpp"

#include "asio/async_result.hpp"
#include "asio/error_code.hpp"

class nullary_completion_handler {
public:
  explicit nullary_completion_handler(bool& invoked) : invoked_(invoked) {}
  void operator()() {
    invoked_ = true;
  }
private:
  bool& invoked_;
};

void redirect_error_nullary_test()
{
  typedef void signature(const asio::error_code&);
  typedef nullary_completion_handler completion_token;
  typedef asio::redirect_error_t<completion_token> redirect_completion_token;
  typedef asio::async_result<redirect_completion_token, signature> async_result;
  bool invoked = false;
  completion_token t(invoked);
  asio::error_code ec;
  async_result::completion_handler_type h(redirect_completion_token(t, ec));
  asio::error_code expected(1, asio::system_category());
  h(expected);
  ASIO_CHECK(invoked);
  ASIO_CHECK(ec == expected);
}

class unary_completion_handler {
public:
  explicit unary_completion_handler(int& i) : i_(i) {}
  void operator()(int i) {
    i_ = i;
  }
private:
  int& i_;
};

void redirect_error_unary_test()
{
  typedef void signature(const asio::error_code&, int);
  typedef unary_completion_handler completion_token;
  typedef asio::redirect_error_t<completion_token> redirect_completion_token;
  typedef asio::async_result<redirect_completion_token, signature> async_result;
  int i = 0;
  completion_token t(i);
  asio::error_code ec;
  async_result::completion_handler_type h(redirect_completion_token(t, ec));
  asio::error_code expected(1, asio::system_category());
  h(expected, 1);
  ASIO_CHECK(i == 1);
  ASIO_CHECK(ec == expected);
}

ASIO_TEST_SUITE
(
  "redirect_error",
  ASIO_TEST_CASE(redirect_error_nullary_test)
  ASIO_TEST_CASE(redirect_error_unary_test)
)
