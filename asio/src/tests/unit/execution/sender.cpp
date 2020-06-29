//
// sender.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2020 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/execution/sender.hpp"

#include "../unit_test.hpp"

namespace exec = asio::execution;

struct not_a_sender
{
};

struct sender_using_base :
  asio::execution::sender_base
{
  sender_using_base()
  {
  }
};

struct executor
{
  executor()
  {
  }

  executor(const executor&) ASIO_NOEXCEPT
  {
  }

#if defined(ASIO_HAS_MOVE)
  executor(executor&&) ASIO_NOEXCEPT
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  template <typename F>
  void execute(ASIO_MOVE_ARG(F) f) const ASIO_NOEXCEPT
  {
    (void)f;
  }

  bool operator==(const executor&) const ASIO_NOEXCEPT
  {
    return true;
  }

  bool operator!=(const executor&) const ASIO_NOEXCEPT
  {
    return false;
  }
};

namespace asio {
namespace traits {

#if !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

template <typename F>
struct execute_member<executor, F>
{
  ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
  ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)
#if !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

template <>
struct equality_comparable<executor>
{
  ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
  ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
};

#endif // !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

} // namespace traits
} // namespace asio

template <typename T>
bool is_unspecialised(T*, ...)
{
  return false;
}

template <typename T>
bool is_unspecialised(T*,
    typename asio::void_type<
      typename exec::sender_traits<
        T>::asio_execution_sender_traits_base_is_unspecialised
    >::type*)
{
  return true;
}

void test_sender_traits()
{
  not_a_sender s1;
  ASIO_CHECK(is_unspecialised(&s1, static_cast<void*>(0)));

  sender_using_base s2;
  ASIO_CHECK(!is_unspecialised(&s2, static_cast<void*>(0)));

  executor s3;
  ASIO_CHECK(!is_unspecialised(&s3, static_cast<void*>(0)));
}

void test_is_sender()
{
  ASIO_CHECK(!exec::is_sender<not_a_sender>::value);
  ASIO_CHECK(exec::is_sender<sender_using_base>::value);
  ASIO_CHECK(exec::is_sender<executor>::value);
}

ASIO_TEST_SUITE
(
  "sender",
  ASIO_TEST_CASE(test_sender_traits)
  ASIO_TEST_CASE(test_is_sender)
)
