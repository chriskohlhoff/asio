//
// any_completion_executor.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#include "asio/any_completion_executor.hpp"

#include <cstring>
#include <functional>
#include <typeinfo>
#include "asio/any_io_executor.hpp"
#include "asio/io_context.hpp"
#include "asio/system_executor.hpp"
#include "asio/thread_pool.hpp"
#include "unit_test.hpp"

using namespace asio;
namespace bindns = std;

static bool next_nothrow_new_fails = false;

void* operator new(std::size_t n,
    const std::nothrow_t&) noexcept
{
  if (next_nothrow_new_fails)
  {
    next_nothrow_new_fails = false;
    return 0;
  }
  return ::operator new(n);
}

struct fat_executor
{
  fat_executor(int id)
    : id_(id)
  {
    std::memset(data_, 0, sizeof(data_));
  }

  template <typename F>
  void execute(const F&) const
  {
  }

  friend bool operator==(const fat_executor& a,
      const fat_executor& b) noexcept
  {
    return a.id_ == b.id_;
  }

  friend bool operator!=(const fat_executor& a,
      const fat_executor& b) noexcept
  {
    return a.id_ != b.id_;
  }

  int id_;
  unsigned char data_[1024];
};

namespace asio {
namespace traits {

#if !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

template <typename F>
struct execute_member<fat_executor, F>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

template <>
struct equality_comparable<fat_executor>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
};

#endif // !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

} // namespace traits
} // namespace asio

void increment(int* count)
{
  ++(*count);
}

void any_completion_executor_construction_test()
{
  thread_pool pool(1);
  asio::nullptr_t null_ptr = asio::nullptr_t();

  asio::any_completion_executor ex1;

  ASIO_CHECK(ex1.target<void>() == 0);
  ASIO_CHECK(ex1 == null_ptr);

  asio::any_completion_executor ex2(null_ptr);

  ASIO_CHECK(ex2.target<void>() == 0);
  ASIO_CHECK(ex2 == null_ptr);
  ASIO_CHECK(ex2 == ex1);

  asio::any_completion_executor ex3(pool.executor());

  ASIO_CHECK(ex3.target<void>() != 0);
  ASIO_CHECK(ex3 != null_ptr);
  ASIO_CHECK(ex3 != ex1);

  asio::any_completion_executor ex4(ex1);

  ASIO_CHECK(ex4.target<void>() == 0);
  ASIO_CHECK(ex4 == null_ptr);
  ASIO_CHECK(ex4 == ex1);

  asio::any_completion_executor ex5(ex3);

  ASIO_CHECK(ex5.target<void>() != 0);
  ASIO_CHECK(ex5 != null_ptr);
  ASIO_CHECK(ex5 == ex3);

  asio::any_completion_executor ex6 = fat_executor(1);

  ASIO_CHECK(ex6.target<void>() != 0);
  ASIO_CHECK(ex6 != null_ptr);
  ASIO_CHECK(ex6 != ex1);

  asio::any_completion_executor ex7 = fat_executor(1);

  ASIO_CHECK(ex7.target<void>() != 0);
  ASIO_CHECK(ex7 != null_ptr);
  ASIO_CHECK(ex7 != ex1);
  ASIO_CHECK(ex7 == ex6);

  asio::any_completion_executor ex8 = fat_executor(2);

  ASIO_CHECK(ex8.target<void>() != 0);
  ASIO_CHECK(ex8 != null_ptr);
  ASIO_CHECK(ex8 != ex1);
  ASIO_CHECK(ex8 != ex6);
  ASIO_CHECK(ex8 != ex7);

  asio::any_completion_executor ex9(ex6);

  ASIO_CHECK(ex9.target<void>() != 0);
  ASIO_CHECK(ex9 != null_ptr);
  ASIO_CHECK(ex9 != ex1);
  ASIO_CHECK(ex9 == ex6);
  ASIO_CHECK(ex9 == ex7);
  ASIO_CHECK(ex9 != ex8);

  asio::any_completion_executor ex10(std::move(ex1));

  ASIO_CHECK(ex10.target<void>() == 0);
  ASIO_CHECK(ex10 == null_ptr);
  ASIO_CHECK(ex1.target<void>() == 0);
  ASIO_CHECK(ex1 == null_ptr);

  asio::any_completion_executor ex11(std::move(ex3));

  ASIO_CHECK(ex11.target<void>() != 0);
  ASIO_CHECK(ex11 != null_ptr);
  ASIO_CHECK(ex3.target<void>() == 0);
  ASIO_CHECK(ex3 == null_ptr);
  ASIO_CHECK(ex11 == ex5);

  asio::any_completion_executor ex12(std::move(ex7));

  ASIO_CHECK(ex12.target<void>() != 0);
  ASIO_CHECK(ex12 != null_ptr);
  ASIO_CHECK(ex7.target<void>() == 0);
  ASIO_CHECK(ex7 == null_ptr);
  ASIO_CHECK(ex12 == ex6);
  ASIO_CHECK(ex12 != ex8);
}

void any_completion_executor_nothrow_construction_test()
{
  thread_pool pool(1);
  asio::nullptr_t null_ptr = asio::nullptr_t();

  asio::any_completion_executor ex1;

  ASIO_CHECK(ex1.target<void>() == 0);
  ASIO_CHECK(ex1 == null_ptr);

  asio::any_completion_executor ex2(null_ptr);

  ASIO_CHECK(ex2.target<void>() == 0);
  ASIO_CHECK(ex2 == null_ptr);
  ASIO_CHECK(ex2 == ex1);

  asio::any_completion_executor ex3(std::nothrow, pool.executor());

  ASIO_CHECK(ex3.target<void>() != 0);
  ASIO_CHECK(ex3 != null_ptr);
  ASIO_CHECK(ex3 != ex1);

  asio::any_completion_executor ex4(std::nothrow, ex1);

  ASIO_CHECK(ex4.target<void>() == 0);
  ASIO_CHECK(ex4 == null_ptr);
  ASIO_CHECK(ex4 == ex1);

  asio::any_completion_executor ex5(std::nothrow, ex3);

  ASIO_CHECK(ex5.target<void>() != 0);
  ASIO_CHECK(ex5 != null_ptr);
  ASIO_CHECK(ex5 == ex3);

  asio::any_completion_executor ex6(std::nothrow, fat_executor(1));

  ASIO_CHECK(ex6.target<void>() != 0);
  ASIO_CHECK(ex6 != null_ptr);
  ASIO_CHECK(ex6 != ex1);

  asio::any_completion_executor ex7(std::nothrow, fat_executor(1));

  ASIO_CHECK(ex7.target<void>() != 0);
  ASIO_CHECK(ex7 != null_ptr);
  ASIO_CHECK(ex7 != ex1);
  ASIO_CHECK(ex7 == ex6);

  asio::any_completion_executor ex8(std::nothrow, fat_executor(2));

  ASIO_CHECK(ex8.target<void>() != 0);
  ASIO_CHECK(ex8 != null_ptr);
  ASIO_CHECK(ex8 != ex1);
  ASIO_CHECK(ex8 != ex6);
  ASIO_CHECK(ex8 != ex7);

  asio::any_completion_executor ex9(std::nothrow, ex6);

  ASIO_CHECK(ex9.target<void>() != 0);
  ASIO_CHECK(ex9 != null_ptr);
  ASIO_CHECK(ex9 != ex1);
  ASIO_CHECK(ex9 == ex6);
  ASIO_CHECK(ex9 == ex7);
  ASIO_CHECK(ex9 != ex8);

  asio::any_completion_executor ex10(std::nothrow, std::move(ex1));

  ASIO_CHECK(ex10.target<void>() == 0);
  ASIO_CHECK(ex10 == null_ptr);
  ASIO_CHECK(ex1.target<void>() == 0);
  ASIO_CHECK(ex1 == null_ptr);

  asio::any_completion_executor ex11(std::nothrow, std::move(ex3));

  ASIO_CHECK(ex11.target<void>() != 0);
  ASIO_CHECK(ex11 != null_ptr);
  ASIO_CHECK(ex3.target<void>() == 0);
  ASIO_CHECK(ex3 == null_ptr);
  ASIO_CHECK(ex11 == ex5);

  asio::any_completion_executor ex12(std::nothrow, std::move(ex7));

  ASIO_CHECK(ex12.target<void>() != 0);
  ASIO_CHECK(ex12 != null_ptr);
  ASIO_CHECK(ex7.target<void>() == 0);
  ASIO_CHECK(ex7 == null_ptr);
  ASIO_CHECK(ex12 == ex6);
  ASIO_CHECK(ex12 != ex8);

  next_nothrow_new_fails = true;
  asio::any_completion_executor ex13(std::nothrow, fat_executor(3));

  ASIO_CHECK(ex13.target<void>() == 0);
  ASIO_CHECK(ex13 == null_ptr);
  ASIO_CHECK(ex13 == ex1);
}

void any_completion_executor_assignment_test()
{
  thread_pool pool(1);
  asio::nullptr_t null_ptr = asio::nullptr_t();

  asio::any_completion_executor ex1;

  asio::any_completion_executor ex2;
  ex2 = null_ptr;

  ASIO_CHECK(ex2.target<void>() == 0);

  asio::any_completion_executor ex3;
  ex3 = pool.executor();

  ASIO_CHECK(ex3.target<void>() != 0);

  asio::any_completion_executor ex4;
  ex4 = ex1;

  ASIO_CHECK(ex4.target<void>() == 0);
  ASIO_CHECK(ex4 == ex1);

  ex4 = ex3;

  ASIO_CHECK(ex4.target<void>() != 0);
  ASIO_CHECK(ex4 == ex3);

  asio::any_completion_executor ex5;
  ex5 = fat_executor(1);

  ASIO_CHECK(ex5.target<void>() != 0);
  ASIO_CHECK(ex5 != null_ptr);
  ASIO_CHECK(ex5 != ex1);

  asio::any_completion_executor ex6;
  ex6 = fat_executor(1);

  ASIO_CHECK(ex6.target<void>() != 0);
  ASIO_CHECK(ex6 != null_ptr);
  ASIO_CHECK(ex6 != ex1);
  ASIO_CHECK(ex6 == ex5);

  ex6 = fat_executor(2);

  ASIO_CHECK(ex6.target<void>() != 0);
  ASIO_CHECK(ex6 != null_ptr);
  ASIO_CHECK(ex6 != ex1);
  ASIO_CHECK(ex6 != ex5);

  asio::any_completion_executor ex7;
  ex7 = ex5;

  ASIO_CHECK(ex7.target<void>() != 0);
  ASIO_CHECK(ex7 != null_ptr);
  ASIO_CHECK(ex7 != ex1);
  ASIO_CHECK(ex7 == ex5);
  ASIO_CHECK(ex7 != ex6);

  asio::any_completion_executor ex8;
  ex8 = std::move(ex1);

  ASIO_CHECK(ex8.target<void>() == 0);
  ASIO_CHECK(ex1.target<void>() == 0);

  ex8 = std::move(ex3);

  ASIO_CHECK(ex8.target<void>() != 0);
  ASIO_CHECK(ex3.target<void>() == 0);
  ASIO_CHECK(ex8 == ex4);

  ex8 = std::move(ex5);

  ASIO_CHECK(ex8.target<void>() != 0);
  ASIO_CHECK(ex5.target<void>() == 0);
  ASIO_CHECK(ex8 == ex7);
}

void any_completion_executor_swap_test()
{
  thread_pool pool1(1);
  thread_pool pool2(1);

  asio::any_completion_executor ex1(pool1.executor());
  asio::any_completion_executor ex2(pool2.executor());

  asio::any_completion_executor ex3(ex1);
  asio::any_completion_executor ex4(ex2);

  ASIO_CHECK(ex3 == ex1);
  ASIO_CHECK(ex4 == ex2);

  ex3.swap(ex4);

  ASIO_CHECK(ex3 == ex2);
  ASIO_CHECK(ex4 == ex1);

  execution::swap(ex3, ex4);

  ASIO_CHECK(ex3 == ex1);
  ASIO_CHECK(ex4 == ex2);
}

void any_completion_executor_query_test()
{
  thread_pool pool(1);
  asio::any_completion_executor ex(pool.executor());

  ASIO_CHECK(
      asio::query(ex, asio::execution::blocking)
        == asio::execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex, asio::execution::blocking.possibly)
        == asio::execution::blocking.possibly);

  ASIO_CHECK(
      asio::query(ex, asio::execution::outstanding_work)
        == asio::execution::outstanding_work.untracked);

  ASIO_CHECK(
      asio::query(ex, asio::execution::outstanding_work.untracked)
        == asio::execution::outstanding_work.untracked);

  ASIO_CHECK(
      asio::query(ex, asio::execution::relationship)
        == asio::execution::relationship.fork);

  ASIO_CHECK(
      asio::query(ex, asio::execution::relationship.fork)
        == asio::execution::relationship.fork);
}

void any_completion_executor_execute_test()
{
  int count = 0;
  thread_pool pool(1);
  asio::any_completion_executor ex(pool.executor());

  ex.execute(bindns::bind(increment, &count));

  asio::prefer(ex, asio::execution::blocking.possibly).execute(
      bindns::bind(increment, &count));

  asio::prefer(ex,
      asio::execution::blocking.possibly,
      asio::execution::outstanding_work.tracked
    ).execute(bindns::bind(increment, &count));

  asio::prefer(ex,
      asio::execution::blocking.possibly,
      asio::execution::outstanding_work.untracked
    ).execute(bindns::bind(increment, &count));

  asio::prefer(ex,
      asio::execution::blocking.possibly,
      asio::execution::outstanding_work.untracked,
      asio::execution::relationship.continuation
    ).execute(bindns::bind(increment, &count));

  pool.wait();

  ASIO_CHECK(count == 5);
}

void any_completion_executor_conversion_test()
{
  io_context ioc1;
  io_context ioc2;
  asio::nullptr_t null_ptr = asio::nullptr_t();
  asio::any_io_executor io1(ioc1.get_executor());
  asio::any_io_executor io2(ioc2.get_executor());
  asio::any_io_executor io3;
  asio::any_completion_executor ex1(io1);

  ASIO_CHECK(ex1.target<void>() != 0);
  ASIO_CHECK(ex1.target<io_context::executor_type>() != 0);
  ASIO_CHECK(*ex1.target<io_context::executor_type>() == ioc1.get_executor());
#if !defined(ASIO_NO_TYPEID)
  ASIO_CHECK(ex1.target_type() == typeid(io_context::executor_type));
#endif // !defined(ASIO_NO_TYPEID)
  ASIO_CHECK(ex1 == asio::any_completion_executor(io1));
  ASIO_CHECK(ex1 == asio::any_completion_executor(ioc1.get_executor()));
  ASIO_CHECK(ex1 != asio::any_completion_executor(io2));

  asio::any_completion_executor ex2(std::nothrow, io2);

  ASIO_CHECK(ex2.target<io_context::executor_type>() != 0);
  ASIO_CHECK(ex2 != ex1);
  ASIO_CHECK(ex2 == asio::any_completion_executor(ioc2.get_executor()));

  asio::any_completion_executor ex3(io3);

  ASIO_CHECK(ex3.target<void>() == 0);
  ASIO_CHECK(ex3 == null_ptr);

  asio::any_completion_executor ex4 = asio::prefer(ex1,
      asio::execution::outstanding_work.tracked);

  ASIO_CHECK(ex4.target<void>() != 0);
  ASIO_CHECK(ex4.target<io_context::executor_type>() == 0);
  ASIO_CHECK(
      asio::query(ex4, asio::execution::outstanding_work)
        == asio::execution::outstanding_work.tracked);
  ASIO_CHECK(
      asio::query(ex1, asio::execution::outstanding_work)
        == asio::execution::outstanding_work.untracked);

  asio::any_completion_executor ex5 = asio::prefer(ex4,
      asio::execution::outstanding_work.untracked);

  ASIO_CHECK(ex5.target<io_context::executor_type>() != 0);
  ASIO_CHECK(
      asio::query(ex5, asio::execution::outstanding_work)
        == asio::execution::outstanding_work.untracked);
  ASIO_CHECK(ex5 == ex1);

  asio::any_completion_executor ex6 = asio::prefer(ex1,
      asio::execution::relationship.continuation);

  ASIO_CHECK(
      asio::query(ex6, asio::execution::relationship)
        == asio::execution::relationship.continuation);
  ASIO_CHECK(
      asio::query(ex1, asio::execution::relationship)
        == asio::execution::relationship.fork);

  typedef execution::any_executor<
      execution::context_as_t<execution_context&>,
      execution::blocking_t::never_t
    > leading_executor;

  leading_executor ex7(io1);

  ASIO_CHECK(ex7.target<io_context::executor_type>() != 0);
  ASIO_CHECK(
      &asio::query(ex7, execution::context_as_t<execution_context&>())
        == &ioc1);
  ASIO_CHECK(
      asio::query(ex7, execution::blocking.never)
        == execution::blocking.possibly);

  leading_executor ex8 = asio::require(ex7, execution::blocking.never);

  ASIO_CHECK(ex8.target<void>() != 0);
  ASIO_CHECK(ex8 != ex7);
  ASIO_CHECK(
      &asio::query(ex8, execution::context_as_t<execution_context&>())
        == &ioc1);
  ASIO_CHECK(
      asio::query(ex8, execution::blocking.never)
        == execution::blocking.never);

  typedef execution::any_executor<
      execution::prefer_only<execution::outstanding_work_t::tracked_t>,
      execution::prefer_only<execution::outstanding_work_t::untracked_t>
    > work_executor;

  work_executor ex9(io1);
  work_executor ex10(ex1);

  ASIO_CHECK(ex9 == ex10);
  ASIO_CHECK(ex10.target<io_context::executor_type>() != 0);

  work_executor ex11 = asio::prefer(ex10,
      asio::execution::outstanding_work.tracked);

  ASIO_CHECK(
      asio::query(ex11, asio::execution::outstanding_work)
        == asio::execution::outstanding_work.tracked);

  work_executor ex12 = asio::prefer(ex11,
      asio::execution::outstanding_work.untracked);

  ASIO_CHECK(ex12 == ex9);

  typedef execution::any_executor<
      execution::blocking_t::never_t,
      execution::prefer_only<execution::relationship_t::continuation_t>
    > other_executor;

  other_executor ex13(io1);

  ASIO_CHECK(ex13.target<io_context::executor_type>() != 0);
#if !defined(ASIO_NO_TYPEID)
  ASIO_CHECK(ex13.target_type() == typeid(io_context::executor_type));
#endif // !defined(ASIO_NO_TYPEID)
  ASIO_CHECK(ex13 == other_executor(io1));
  ASIO_CHECK(ex13 != other_executor(io2));
  ASIO_CHECK(
      asio::query(ex13, execution::relationship)
        == execution::relationship.fork);

  other_executor ex14 = asio::prefer(ex13,
      execution::relationship.continuation);

  ASIO_CHECK(
      asio::query(ex14, execution::relationship)
        == execution::relationship.continuation);

  other_executor ex15 = asio::require(ex13, execution::blocking.never);

  ASIO_CHECK(
      asio::query(ex15, execution::blocking.never)
        == execution::blocking.never);

  int count = 0;
  ex1.execute(bindns::bind(increment, &count));
  ex8.execute(bindns::bind(increment, &count));
  ex11.execute(bindns::bind(increment, &count));
  ex15.execute(bindns::bind(increment, &count));
  ioc1.poll();

  ASIO_CHECK(count == 4);
}

ASIO_TEST_SUITE
(
  "any_completion_executor",
  ASIO_TEST_CASE(any_completion_executor_construction_test)
  ASIO_TEST_CASE(any_completion_executor_nothrow_construction_test)
  ASIO_TEST_CASE(any_completion_executor_assignment_test)
  ASIO_TEST_CASE(any_completion_executor_swap_test)
  ASIO_TEST_CASE(any_completion_executor_query_test)
  ASIO_TEST_CASE(any_completion_executor_execute_test)
  ASIO_TEST_CASE(any_completion_executor_conversion_test)
)
