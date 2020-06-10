//
// prefer_only.cpp
// ~~~~~~~~~~~~~~~
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
#include "asio/execution/prefer_only.hpp"

#include "asio/execution/any_executor.hpp"
#include "../unit_test.hpp"

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

static int possibly_blocking_count = 0;
static int never_blocking_count = 0;

struct possibly_blocking_executor
{
  template <typename F>
  void execute(const F&) const
  {
    ++possibly_blocking_count;
  }

  friend bool operator==(const possibly_blocking_executor&,
      const possibly_blocking_executor&) ASIO_NOEXCEPT
  {
    return true;
  }

  friend bool operator!=(const possibly_blocking_executor&,
      const possibly_blocking_executor&) ASIO_NOEXCEPT
  {
    return false;
  }
};

#if !defined(ASIO_HAS_DEDUCED_EXECUTION_IS_EXECUTOR_TRAIT)

namespace asio {
namespace execution {

template <>
struct is_executor<possibly_blocking_executor> : asio::true_type
{
};

} // namespace execution
} // namespace asio

#endif // !defined(ASIO_HAS_DEDUCED_EXECUTION_IS_EXECUTOR_TRAIT)

struct never_blocking_executor
{
  static ASIO_CONSTEXPR execution::blocking_t::never_t
    query(execution::blocking_t) ASIO_NOEXCEPT
  {
    return execution::blocking_t::never_t();
  }

  template <typename F>
  void execute(const F&) const
  {
    ++never_blocking_count;
  }

  friend bool operator==(const never_blocking_executor&,
      const never_blocking_executor&) ASIO_NOEXCEPT
  {
    return true;
  }

  friend bool operator!=(const never_blocking_executor&,
      const never_blocking_executor&) ASIO_NOEXCEPT
  {
    return false;
  }
};

#if !defined(ASIO_HAS_DEDUCED_EXECUTION_IS_EXECUTOR_TRAIT)

namespace asio {
namespace execution {

template <>
struct is_executor<never_blocking_executor> : asio::true_type
{
};

} // namespace execution
} // namespace asio

#endif // !defined(ASIO_HAS_DEDUCED_EXECUTION_IS_EXECUTOR_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_QUERY_STATIC_CONSTEXPR_MEMBER_TRAIT)

namespace asio {
namespace traits {

template <typename Param>
struct query_static_constexpr_member<
  never_blocking_executor, Param,
  typename asio::enable_if<
    asio::is_convertible<Param, execution::blocking_t>::value
  >::type>
{
  ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
  ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

  typedef execution::blocking_t::never_t result_type;

  static ASIO_CONSTEXPR result_type value()
  {
    return result_type();
  }
};

} // namespace traits
} // namespace asio

#endif // !defined(ASIO_HAS_DEDUCED_QUERY_STATIC_CONSTEXPR_MEMBER_TRAIT)

struct either_blocking_executor
{
  execution::blocking_t blocking_;

  explicit either_blocking_executor(execution::blocking_t b)
    : blocking_(b)
  {
  }

  execution::blocking_t query(execution::blocking_t) const ASIO_NOEXCEPT
  {
    return blocking_;
  }

  either_blocking_executor require(execution::blocking_t::possibly_t) const
  {
    return either_blocking_executor(execution::blocking.possibly);
  }

  either_blocking_executor require(execution::blocking_t::never_t) const
  {
    return either_blocking_executor(execution::blocking.never);
  }

  template <typename F>
  void execute(const F&) const
  {
    if (blocking_ == execution::blocking.never)
      ++never_blocking_count;
    else
      ++possibly_blocking_count;
  }

  friend bool operator==(const either_blocking_executor& a,
      const either_blocking_executor& b) ASIO_NOEXCEPT
  {
    return a.blocking_ == b.blocking_;
  }

  friend bool operator!=(const either_blocking_executor& a,
      const either_blocking_executor& b) ASIO_NOEXCEPT
  {
    return a.blocking_ != b.blocking_;
  }
};

#if !defined(ASIO_HAS_DEDUCED_EXECUTION_IS_EXECUTOR_TRAIT)

namespace asio {
namespace execution {

template <>
struct is_executor<either_blocking_executor> : asio::true_type
{
};

} // namespace execution
} // namespace asio

#endif // !defined(ASIO_HAS_DEDUCED_EXECUTION_IS_EXECUTOR_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

namespace asio {
namespace traits {

template <typename Param>
struct query_member<
  either_blocking_executor, Param,
  typename asio::enable_if<
    asio::is_convertible<Param, execution::blocking_t>::value
  >::type>
{
  ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
  ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

  typedef execution::blocking_t result_type;
};

} // namespace traits
} // namespace asio

#endif // !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

namespace asio {
namespace traits {

template <typename Param>
struct require_member<
  either_blocking_executor, Param,
  typename asio::enable_if<
    asio::is_convertible<Param, execution::blocking_t>::value
  >::type>
{
  ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
  ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);

  typedef either_blocking_executor result_type;
};

} // namespace traits
} // namespace asio

#endif // !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

void prefer_only_executor_query_test()
{
  typedef execution::any_executor<
      execution::blocking_t,
      execution::prefer_only<execution::blocking_t::possibly_t>,
      execution::prefer_only<execution::blocking_t::never_t>
    > executor_type;

  executor_type ex1 = possibly_blocking_executor();

  ASIO_CHECK(
      asio::query(ex1, execution::blocking)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex1, execution::blocking.possibly)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex1, execution::blocking.never)
        == execution::blocking.possibly);

  executor_type ex2 = asio::prefer(ex1, execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex2, execution::blocking)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex2, execution::blocking.possibly)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex2, execution::blocking.never)
        == execution::blocking.possibly);

  executor_type ex3 = asio::prefer(ex1, execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex3, execution::blocking)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex3, execution::blocking.possibly)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex3, execution::blocking.never)
        == execution::blocking.possibly);

  executor_type ex4 = never_blocking_executor();

  ASIO_CHECK(
      asio::query(ex4, execution::blocking)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex4, execution::blocking.possibly)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex4, execution::blocking.never)
        == execution::blocking.never);

  executor_type ex5 = asio::prefer(ex4, execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex5, execution::blocking)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex5, execution::blocking.possibly)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex5, execution::blocking.never)
        == execution::blocking.never);

  executor_type ex6 = asio::prefer(ex4, execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex6, execution::blocking)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex6, execution::blocking.possibly)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex6, execution::blocking.never)
        == execution::blocking.never);

  executor_type ex7 = either_blocking_executor(execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex7, execution::blocking)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex7, execution::blocking.possibly)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex7, execution::blocking.never)
        == execution::blocking.possibly);

  executor_type ex8 = asio::prefer(ex7, execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex8, execution::blocking)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex8, execution::blocking.possibly)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex8, execution::blocking.never)
        == execution::blocking.possibly);

  executor_type ex9 = asio::prefer(ex7, execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex9, execution::blocking)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex9, execution::blocking.possibly)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex9, execution::blocking.never)
        == execution::blocking.never);

  executor_type ex10 = either_blocking_executor(execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex10, execution::blocking)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex10, execution::blocking.possibly)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex10, execution::blocking.never)
        == execution::blocking.never);

  executor_type ex11 = asio::prefer(ex7, execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex11, execution::blocking)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex11, execution::blocking.possibly)
        == execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex11, execution::blocking.never)
        == execution::blocking.possibly);

  executor_type ex12 = asio::prefer(ex7, execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex12, execution::blocking)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex12, execution::blocking.possibly)
        == execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex12, execution::blocking.never)
        == execution::blocking.never);
}

void do_nothing()
{
}

void prefer_only_executor_execute_test()
{
  typedef execution::any_executor<
      execution::blocking_t,
      execution::prefer_only<execution::blocking_t::possibly_t>,
      execution::prefer_only<execution::blocking_t::never_t>
    > executor_type;

  executor_type ex1 = possibly_blocking_executor();

  execution::execute(ex1, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 1);
  ASIO_CHECK(never_blocking_count == 0);

  executor_type ex2 = asio::prefer(ex1, execution::blocking.possibly);

  execution::execute(ex2, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 2);
  ASIO_CHECK(never_blocking_count == 0);

  executor_type ex3 = asio::prefer(ex1, execution::blocking.never);

  execution::execute(ex3, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 3);
  ASIO_CHECK(never_blocking_count == 0);

  executor_type ex4 = never_blocking_executor();

  execution::execute(ex4, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 3);
  ASIO_CHECK(never_blocking_count == 1);

  executor_type ex5 = asio::prefer(ex4, execution::blocking.possibly);

  execution::execute(ex5, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 3);
  ASIO_CHECK(never_blocking_count == 2);

  executor_type ex6 = asio::prefer(ex4, execution::blocking.never);

  execution::execute(ex6, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 3);
  ASIO_CHECK(never_blocking_count == 3);

  executor_type ex7 = either_blocking_executor(execution::blocking.possibly);

  execution::execute(ex7, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 4);
  ASIO_CHECK(never_blocking_count == 3);

  executor_type ex8 = asio::prefer(ex7, execution::blocking.possibly);

  execution::execute(ex8, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 5);
  ASIO_CHECK(never_blocking_count == 3);

  executor_type ex9 = asio::prefer(ex7, execution::blocking.never);

  execution::execute(ex9, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 5);
  ASIO_CHECK(never_blocking_count == 4);

  executor_type ex10 = either_blocking_executor(execution::blocking.never);

  execution::execute(ex10, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 5);
  ASIO_CHECK(never_blocking_count == 5);

  executor_type ex11 = asio::prefer(ex7, execution::blocking.possibly);

  execution::execute(ex11, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 6);
  ASIO_CHECK(never_blocking_count == 5);

  executor_type ex12 = asio::prefer(ex7, execution::blocking.never);

  execution::execute(ex12, &do_nothing);
  ASIO_CHECK(possibly_blocking_count == 6);
  ASIO_CHECK(never_blocking_count == 6);
}

ASIO_TEST_SUITE
(
  "prefer_only",
  ASIO_TEST_CASE(prefer_only_executor_query_test)
  ASIO_TEST_CASE(prefer_only_executor_execute_test)
)
