//
// any_executor.cpp
// ~~~~~~~~~~~~~~~~
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
#include "asio/execution/any_executor.hpp"

#include <cstring>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include "asio/thread_pool.hpp"
#include "asio/io_context.hpp"
#include "asio/any_completion_executor.hpp"
#include "asio/any_io_executor.hpp"
#include "asio/detail/executor_function.hpp"
#include "asio/strand.hpp"
#include "../unit_test.hpp"

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

  std::size_t query(execution::occupancy_t) const
  {
    return 1;
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
  static constexpr bool is_noexcept = true;
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

template <>
struct query_member<fat_executor, execution::occupancy_t>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef std::size_t result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

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

void any_executor_construction_test()
{
  typedef execution::any_executor<> ex_no_props_t;

  typedef execution::any_executor<
      execution::blocking_t
    > ex_one_prop_t;

  typedef execution::any_executor<
      execution::blocking_t,
      execution::occupancy_t
    > ex_two_props_t;

  thread_pool pool(1);
  asio::nullptr_t null_ptr = asio::nullptr_t();

  ex_two_props_t ex_two_props_1;

  ASIO_CHECK(ex_two_props_1.target<void>() == 0);
  ASIO_CHECK(ex_two_props_1 == null_ptr);

  ex_two_props_t ex_two_props_2(null_ptr);

  ASIO_CHECK(ex_two_props_2.target<void>() == 0);
  ASIO_CHECK(ex_two_props_2 == null_ptr);
  ASIO_CHECK(ex_two_props_2 == ex_two_props_1);

  ex_two_props_t ex_two_props_3(pool.executor());

  ASIO_CHECK(ex_two_props_3.target<void>() != 0);
  ASIO_CHECK(ex_two_props_3 != null_ptr);
  ASIO_CHECK(ex_two_props_3 != ex_two_props_1);

  ex_two_props_t ex_two_props_4(ex_two_props_1);

  ASIO_CHECK(ex_two_props_4.target<void>() == 0);
  ASIO_CHECK(ex_two_props_4 == null_ptr);
  ASIO_CHECK(ex_two_props_4 == ex_two_props_1);

  ex_two_props_t ex_two_props_5(ex_two_props_3);

  ASIO_CHECK(ex_two_props_5.target<void>() != 0);
  ASIO_CHECK(ex_two_props_5 != null_ptr);
  ASIO_CHECK(ex_two_props_5 == ex_two_props_3);

  ex_two_props_t ex_two_props_6 = fat_executor(1);

  ASIO_CHECK(ex_two_props_6.target<void>() != 0);
  ASIO_CHECK(ex_two_props_6 != null_ptr);
  ASIO_CHECK(ex_two_props_6 != ex_two_props_1);

  ex_two_props_t ex_two_props_7 = fat_executor(1);

  ASIO_CHECK(ex_two_props_7.target<void>() != 0);
  ASIO_CHECK(ex_two_props_7 != null_ptr);
  ASIO_CHECK(ex_two_props_7 != ex_two_props_1);
  ASIO_CHECK(ex_two_props_7 == ex_two_props_6);

  ex_two_props_t ex_two_props_8 = fat_executor(2);

  ASIO_CHECK(ex_two_props_8.target<void>() != 0);
  ASIO_CHECK(ex_two_props_8 != null_ptr);
  ASIO_CHECK(ex_two_props_8 != ex_two_props_1);
  ASIO_CHECK(ex_two_props_8 != ex_two_props_6);
  ASIO_CHECK(ex_two_props_8 != ex_two_props_7);

  ex_two_props_t ex_two_props_9(ex_two_props_6);

  ASIO_CHECK(ex_two_props_9.target<void>() != 0);
  ASIO_CHECK(ex_two_props_9 != null_ptr);
  ASIO_CHECK(ex_two_props_9 != ex_two_props_1);
  ASIO_CHECK(ex_two_props_9 == ex_two_props_6);
  ASIO_CHECK(ex_two_props_9 == ex_two_props_7);
  ASIO_CHECK(ex_two_props_9 != ex_two_props_8);

  ex_two_props_t ex_two_props_10(std::move(ex_two_props_1));

  ASIO_CHECK(ex_two_props_10.target<void>() == 0);
  ASIO_CHECK(ex_two_props_10 == null_ptr);
  ASIO_CHECK(ex_two_props_1.target<void>() == 0);
  ASIO_CHECK(ex_two_props_1 == null_ptr);

  ex_two_props_t ex_two_props_11(std::move(ex_two_props_3));

  ASIO_CHECK(ex_two_props_11.target<void>() != 0);
  ASIO_CHECK(ex_two_props_11 != null_ptr);
  ASIO_CHECK(ex_two_props_3.target<void>() == 0);
  ASIO_CHECK(ex_two_props_3 == null_ptr);
  ASIO_CHECK(ex_two_props_11 == ex_two_props_5);

  ex_two_props_t ex_two_props_12(std::move(ex_two_props_7));

  ASIO_CHECK(ex_two_props_12.target<void>() != 0);
  ASIO_CHECK(ex_two_props_12 != null_ptr);
  ASIO_CHECK(ex_two_props_7.target<void>() == 0);
  ASIO_CHECK(ex_two_props_7 == null_ptr);
  ASIO_CHECK(ex_two_props_12 == ex_two_props_6);
  ASIO_CHECK(ex_two_props_12 != ex_two_props_8);

  ex_one_prop_t ex_one_prop_1;

  ASIO_CHECK(ex_one_prop_1.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_1 == null_ptr);

  ex_one_prop_t ex_one_prop_2(null_ptr);

  ASIO_CHECK(ex_one_prop_2.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_2 == null_ptr);
  ASIO_CHECK(ex_one_prop_2 == ex_one_prop_1);

  ex_one_prop_t ex_one_prop_3(pool.executor());

  ASIO_CHECK(ex_one_prop_3.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_3 != null_ptr);
  ASIO_CHECK(ex_one_prop_3 != ex_one_prop_1);

  ex_one_prop_t ex_one_prop_4(ex_one_prop_1);

  ASIO_CHECK(ex_one_prop_4.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_4 == null_ptr);
  ASIO_CHECK(ex_one_prop_4 == ex_one_prop_1);

  ex_one_prop_t ex_one_prop_5(ex_one_prop_3);

  ASIO_CHECK(ex_one_prop_5.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_5 != null_ptr);
  ASIO_CHECK(ex_one_prop_5 == ex_one_prop_3);

  ex_one_prop_t ex_one_prop_6 = fat_executor(1);

  ASIO_CHECK(ex_one_prop_6.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_6 != null_ptr);
  ASIO_CHECK(ex_one_prop_6 != ex_one_prop_1);

  ex_one_prop_t ex_one_prop_7 = fat_executor(1);

  ASIO_CHECK(ex_one_prop_7.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_7 != null_ptr);
  ASIO_CHECK(ex_one_prop_7 != ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_7 == ex_one_prop_6);

  ex_one_prop_t ex_one_prop_8 = fat_executor(2);

  ASIO_CHECK(ex_one_prop_8.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_8 != null_ptr);
  ASIO_CHECK(ex_one_prop_8 != ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_8 != ex_one_prop_6);
  ASIO_CHECK(ex_one_prop_8 != ex_one_prop_7);

  ex_one_prop_t ex_one_prop_9(ex_one_prop_6);

  ASIO_CHECK(ex_one_prop_9.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_9 != null_ptr);
  ASIO_CHECK(ex_one_prop_9 != ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_9 == ex_one_prop_6);
  ASIO_CHECK(ex_one_prop_9 == ex_one_prop_7);
  ASIO_CHECK(ex_one_prop_9 != ex_one_prop_8);

  ex_one_prop_t ex_one_prop_10(std::move(ex_one_prop_1));

  ASIO_CHECK(ex_one_prop_10.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_10 == null_ptr);
  ASIO_CHECK(ex_one_prop_1.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_1 == null_ptr);

  ex_one_prop_t ex_one_prop_11(std::move(ex_one_prop_3));

  ASIO_CHECK(ex_one_prop_11.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_11 != null_ptr);
  ASIO_CHECK(ex_one_prop_3.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_3 == null_ptr);
  ASIO_CHECK(ex_one_prop_11 == ex_one_prop_5);

  ex_one_prop_t ex_one_prop_12(std::move(ex_one_prop_7));

  ASIO_CHECK(ex_one_prop_12.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_12 != null_ptr);
  ASIO_CHECK(ex_one_prop_7.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_7 == null_ptr);
  ASIO_CHECK(ex_one_prop_12 == ex_one_prop_6);
  ASIO_CHECK(ex_one_prop_12 != ex_one_prop_8);

  ex_one_prop_t ex_one_prop_13(ex_two_props_1);

  ASIO_CHECK(ex_one_prop_13.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_13 == null_ptr);

  ex_one_prop_t ex_one_prop_14(ex_two_props_5);

  ASIO_CHECK(ex_one_prop_14.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_14 != null_ptr);

  ex_one_prop_t ex_one_prop_15(ex_two_props_9);

  ASIO_CHECK(ex_one_prop_15.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_15 != null_ptr);

  ex_no_props_t ex_no_props_1;

  ASIO_CHECK(ex_no_props_1.target<void>() == 0);
  ASIO_CHECK(ex_no_props_1 == null_ptr);

  ex_no_props_t ex_no_props_2(null_ptr);

  ASIO_CHECK(ex_no_props_2.target<void>() == 0);
  ASIO_CHECK(ex_no_props_2 == null_ptr);
  ASIO_CHECK(ex_no_props_2 == ex_no_props_1);

  ex_no_props_t ex_no_props_3(pool.executor());

  ASIO_CHECK(ex_no_props_3.target<void>() != 0);
  ASIO_CHECK(ex_no_props_3 != null_ptr);
  ASIO_CHECK(ex_no_props_3 != ex_no_props_1);

  ex_no_props_t ex_no_props_4(ex_no_props_1);

  ASIO_CHECK(ex_no_props_4.target<void>() == 0);
  ASIO_CHECK(ex_no_props_4 == null_ptr);
  ASIO_CHECK(ex_no_props_4 == ex_no_props_1);

  ex_no_props_t ex_no_props_5(ex_no_props_3);

  ASIO_CHECK(ex_no_props_5.target<void>() != 0);
  ASIO_CHECK(ex_no_props_5 != null_ptr);
  ASIO_CHECK(ex_no_props_5 == ex_no_props_3);

  ex_no_props_t ex_no_props_6 = fat_executor(1);

  ASIO_CHECK(ex_no_props_6.target<void>() != 0);
  ASIO_CHECK(ex_no_props_6 != null_ptr);
  ASIO_CHECK(ex_no_props_6 != ex_no_props_1);

  ex_no_props_t ex_no_props_7 = fat_executor(1);

  ASIO_CHECK(ex_no_props_7.target<void>() != 0);
  ASIO_CHECK(ex_no_props_7 != null_ptr);
  ASIO_CHECK(ex_no_props_7 != ex_no_props_1);
  ASIO_CHECK(ex_no_props_7 == ex_no_props_6);

  ex_no_props_t ex_no_props_8 = fat_executor(2);

  ASIO_CHECK(ex_no_props_8.target<void>() != 0);
  ASIO_CHECK(ex_no_props_8 != null_ptr);
  ASIO_CHECK(ex_no_props_8 != ex_no_props_1);
  ASIO_CHECK(ex_no_props_8 != ex_no_props_6);
  ASIO_CHECK(ex_no_props_8 != ex_no_props_7);

  ex_no_props_t ex_no_props_9(ex_no_props_6);

  ASIO_CHECK(ex_no_props_9.target<void>() != 0);
  ASIO_CHECK(ex_no_props_9 != null_ptr);
  ASIO_CHECK(ex_no_props_9 != ex_no_props_1);
  ASIO_CHECK(ex_no_props_9 == ex_no_props_6);
  ASIO_CHECK(ex_no_props_9 == ex_no_props_7);
  ASIO_CHECK(ex_no_props_9 != ex_no_props_8);

  ex_no_props_t ex_no_props_10(std::move(ex_no_props_1));

  ASIO_CHECK(ex_no_props_10.target<void>() == 0);
  ASIO_CHECK(ex_no_props_10 == null_ptr);
  ASIO_CHECK(ex_no_props_1.target<void>() == 0);
  ASIO_CHECK(ex_no_props_1 == null_ptr);

  ex_no_props_t ex_no_props_11(std::move(ex_no_props_3));

  ASIO_CHECK(ex_no_props_11.target<void>() != 0);
  ASIO_CHECK(ex_no_props_11 != null_ptr);
  ASIO_CHECK(ex_no_props_3.target<void>() == 0);
  ASIO_CHECK(ex_no_props_3 == null_ptr);
  ASIO_CHECK(ex_no_props_11 == ex_no_props_5);

  ex_no_props_t ex_no_props_12(std::move(ex_no_props_7));

  ASIO_CHECK(ex_no_props_12.target<void>() != 0);
  ASIO_CHECK(ex_no_props_12 != null_ptr);
  ASIO_CHECK(ex_no_props_7.target<void>() == 0);
  ASIO_CHECK(ex_no_props_7 == null_ptr);
  ASIO_CHECK(ex_no_props_12 == ex_no_props_6);
  ASIO_CHECK(ex_no_props_12 != ex_no_props_8);

  ex_no_props_t ex_no_props_13(ex_two_props_1);

  ASIO_CHECK(ex_no_props_13.target<void>() == 0);
  ASIO_CHECK(ex_no_props_13 == null_ptr);

  ex_no_props_t ex_no_props_14(ex_two_props_5);

  ASIO_CHECK(ex_no_props_14.target<void>() != 0);
  ASIO_CHECK(ex_no_props_14 != null_ptr);

  ex_no_props_t ex_no_props_15(ex_two_props_9);

  ASIO_CHECK(ex_no_props_15.target<void>() != 0);
  ASIO_CHECK(ex_no_props_15 != null_ptr);

  ex_no_props_t ex_no_props_16(ex_one_prop_1);

  ASIO_CHECK(ex_no_props_16.target<void>() == 0);
  ASIO_CHECK(ex_no_props_16 == null_ptr);

  ex_no_props_t ex_no_props_17(ex_one_prop_5);

  ASIO_CHECK(ex_no_props_17.target<void>() != 0);
  ASIO_CHECK(ex_no_props_17 != null_ptr);

  ex_no_props_t ex_no_props_18(ex_one_prop_9);

  ASIO_CHECK(ex_no_props_18.target<void>() != 0);
  ASIO_CHECK(ex_no_props_18 != null_ptr);
}

void any_executor_nothrow_construction_test()
{
  typedef execution::any_executor<> ex_no_props_t;

  typedef execution::any_executor<
      execution::blocking_t
    > ex_one_prop_t;

  typedef execution::any_executor<
      execution::blocking_t,
      execution::occupancy_t
    > ex_two_props_t;

  thread_pool pool(1);
  asio::nullptr_t null_ptr = asio::nullptr_t();

  ex_two_props_t ex_two_props_1;

  ASIO_CHECK(ex_two_props_1.target<void>() == 0);
  ASIO_CHECK(ex_two_props_1 == null_ptr);

  ex_two_props_t ex_two_props_2(null_ptr);

  ASIO_CHECK(ex_two_props_2.target<void>() == 0);
  ASIO_CHECK(ex_two_props_2 == null_ptr);
  ASIO_CHECK(ex_two_props_2 == ex_two_props_1);

  ex_two_props_t ex_two_props_3(std::nothrow, pool.executor());

  ASIO_CHECK(ex_two_props_3.target<void>() != 0);
  ASIO_CHECK(ex_two_props_3 != null_ptr);
  ASIO_CHECK(ex_two_props_3 != ex_two_props_1);

  ex_two_props_t ex_two_props_4(std::nothrow, ex_two_props_1);

  ASIO_CHECK(ex_two_props_4.target<void>() == 0);
  ASIO_CHECK(ex_two_props_4 == null_ptr);
  ASIO_CHECK(ex_two_props_4 == ex_two_props_1);

  ex_two_props_t ex_two_props_5(std::nothrow, ex_two_props_3);

  ASIO_CHECK(ex_two_props_5.target<void>() != 0);
  ASIO_CHECK(ex_two_props_5 != null_ptr);
  ASIO_CHECK(ex_two_props_5 == ex_two_props_3);

  ex_two_props_t ex_two_props_6(std::nothrow, fat_executor(1));

  ASIO_CHECK(ex_two_props_6.target<void>() != 0);
  ASIO_CHECK(ex_two_props_6 != null_ptr);
  ASIO_CHECK(ex_two_props_6 != ex_two_props_1);

  ex_two_props_t ex_two_props_7(std::nothrow, fat_executor(1));

  ASIO_CHECK(ex_two_props_7.target<void>() != 0);
  ASIO_CHECK(ex_two_props_7 != null_ptr);
  ASIO_CHECK(ex_two_props_7 != ex_two_props_1);
  ASIO_CHECK(ex_two_props_7 == ex_two_props_6);

  ex_two_props_t ex_two_props_8(std::nothrow, fat_executor(2));

  ASIO_CHECK(ex_two_props_8.target<void>() != 0);
  ASIO_CHECK(ex_two_props_8 != null_ptr);
  ASIO_CHECK(ex_two_props_8 != ex_two_props_1);
  ASIO_CHECK(ex_two_props_8 != ex_two_props_6);
  ASIO_CHECK(ex_two_props_8 != ex_two_props_7);

  ex_two_props_t ex_two_props_9(std::nothrow, ex_two_props_6);

  ASIO_CHECK(ex_two_props_9.target<void>() != 0);
  ASIO_CHECK(ex_two_props_9 != null_ptr);
  ASIO_CHECK(ex_two_props_9 != ex_two_props_1);
  ASIO_CHECK(ex_two_props_9 == ex_two_props_6);
  ASIO_CHECK(ex_two_props_9 == ex_two_props_7);
  ASIO_CHECK(ex_two_props_9 != ex_two_props_8);

  ex_two_props_t ex_two_props_10(std::nothrow, std::move(ex_two_props_1));

  ASIO_CHECK(ex_two_props_10.target<void>() == 0);
  ASIO_CHECK(ex_two_props_10 == null_ptr);
  ASIO_CHECK(ex_two_props_1.target<void>() == 0);
  ASIO_CHECK(ex_two_props_1 == null_ptr);

  ex_two_props_t ex_two_props_11(std::nothrow, std::move(ex_two_props_3));

  ASIO_CHECK(ex_two_props_11.target<void>() != 0);
  ASIO_CHECK(ex_two_props_11 != null_ptr);
  ASIO_CHECK(ex_two_props_3.target<void>() == 0);
  ASIO_CHECK(ex_two_props_3 == null_ptr);
  ASIO_CHECK(ex_two_props_11 == ex_two_props_5);

  ex_two_props_t ex_two_props_12(std::nothrow, std::move(ex_two_props_7));

  ASIO_CHECK(ex_two_props_12.target<void>() != 0);
  ASIO_CHECK(ex_two_props_12 != null_ptr);
  ASIO_CHECK(ex_two_props_7.target<void>() == 0);
  ASIO_CHECK(ex_two_props_7 == null_ptr);
  ASIO_CHECK(ex_two_props_12 == ex_two_props_6);
  ASIO_CHECK(ex_two_props_12 != ex_two_props_8);

  next_nothrow_new_fails = true;
  ex_two_props_t ex_two_props_13(std::nothrow, fat_executor(3));

  ASIO_CHECK(ex_two_props_13.target<void>() == 0);
  ASIO_CHECK(ex_two_props_13 == null_ptr);
  ASIO_CHECK(ex_two_props_13 == ex_two_props_1);

  ex_one_prop_t ex_one_prop_1;

  ASIO_CHECK(ex_one_prop_1.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_1 == null_ptr);

  ex_one_prop_t ex_one_prop_2(null_ptr);

  ASIO_CHECK(ex_one_prop_2.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_2 == null_ptr);
  ASIO_CHECK(ex_one_prop_2 == ex_one_prop_1);

  ex_one_prop_t ex_one_prop_3(std::nothrow, pool.executor());

  ASIO_CHECK(ex_one_prop_3.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_3 != null_ptr);
  ASIO_CHECK(ex_one_prop_3 != ex_one_prop_1);

  ex_one_prop_t ex_one_prop_4(std::nothrow, ex_one_prop_1);

  ASIO_CHECK(ex_one_prop_4.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_4 == null_ptr);
  ASIO_CHECK(ex_one_prop_4 == ex_one_prop_1);

  ex_one_prop_t ex_one_prop_5(std::nothrow, ex_one_prop_3);

  ASIO_CHECK(ex_one_prop_5.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_5 != null_ptr);
  ASIO_CHECK(ex_one_prop_5 == ex_one_prop_3);

  ex_one_prop_t ex_one_prop_6 = fat_executor(1);

  ASIO_CHECK(ex_one_prop_6.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_6 != null_ptr);
  ASIO_CHECK(ex_one_prop_6 != ex_one_prop_1);

  ex_one_prop_t ex_one_prop_7(std::nothrow, fat_executor(1));

  ASIO_CHECK(ex_one_prop_7.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_7 != null_ptr);
  ASIO_CHECK(ex_one_prop_7 != ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_7 == ex_one_prop_6);

  ex_one_prop_t ex_one_prop_8(std::nothrow, fat_executor(2));

  ASIO_CHECK(ex_one_prop_8.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_8 != null_ptr);
  ASIO_CHECK(ex_one_prop_8 != ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_8 != ex_one_prop_6);
  ASIO_CHECK(ex_one_prop_8 != ex_one_prop_7);

  ex_one_prop_t ex_one_prop_9(std::nothrow, ex_one_prop_6);

  ASIO_CHECK(ex_one_prop_9.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_9 != null_ptr);
  ASIO_CHECK(ex_one_prop_9 != ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_9 == ex_one_prop_6);
  ASIO_CHECK(ex_one_prop_9 == ex_one_prop_7);
  ASIO_CHECK(ex_one_prop_9 != ex_one_prop_8);

  ex_one_prop_t ex_one_prop_10(std::nothrow, std::move(ex_one_prop_1));

  ASIO_CHECK(ex_one_prop_10.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_10 == null_ptr);
  ASIO_CHECK(ex_one_prop_1.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_1 == null_ptr);

  ex_one_prop_t ex_one_prop_11(std::nothrow, std::move(ex_one_prop_3));

  ASIO_CHECK(ex_one_prop_11.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_11 != null_ptr);
  ASIO_CHECK(ex_one_prop_3.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_3 == null_ptr);
  ASIO_CHECK(ex_one_prop_11 == ex_one_prop_5);

  ex_one_prop_t ex_one_prop_12(std::nothrow, std::move(ex_one_prop_7));

  ASIO_CHECK(ex_one_prop_12.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_12 != null_ptr);
  ASIO_CHECK(ex_one_prop_7.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_7 == null_ptr);
  ASIO_CHECK(ex_one_prop_12 == ex_one_prop_6);
  ASIO_CHECK(ex_one_prop_12 != ex_one_prop_8);

  ex_one_prop_t ex_one_prop_13(std::nothrow, ex_two_props_1);

  ASIO_CHECK(ex_one_prop_13.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_13 == null_ptr);

  ex_one_prop_t ex_one_prop_14(std::nothrow, ex_two_props_5);

  ASIO_CHECK(ex_one_prop_14.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_14 != null_ptr);

  ex_one_prop_t ex_one_prop_15(std::nothrow, ex_two_props_9);

  ASIO_CHECK(ex_one_prop_15.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_15 != null_ptr);

  next_nothrow_new_fails = true;
  ex_one_prop_t ex_one_prop_16(std::nothrow, fat_executor(3));

  ASIO_CHECK(ex_one_prop_16.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_16 == null_ptr);
  ASIO_CHECK(ex_one_prop_16 == ex_one_prop_1);

  ex_no_props_t ex_no_props_1;

  ASIO_CHECK(ex_no_props_1.target<void>() == 0);
  ASIO_CHECK(ex_no_props_1 == null_ptr);

  ex_no_props_t ex_no_props_2(null_ptr);

  ASIO_CHECK(ex_no_props_2.target<void>() == 0);
  ASIO_CHECK(ex_no_props_2 == null_ptr);
  ASIO_CHECK(ex_no_props_2 == ex_no_props_1);

  ex_no_props_t ex_no_props_3(std::nothrow, pool.executor());

  ASIO_CHECK(ex_no_props_3.target<void>() != 0);
  ASIO_CHECK(ex_no_props_3 != null_ptr);
  ASIO_CHECK(ex_no_props_3 != ex_no_props_1);

  ex_no_props_t ex_no_props_4(std::nothrow, ex_no_props_1);

  ASIO_CHECK(ex_no_props_4.target<void>() == 0);
  ASIO_CHECK(ex_no_props_4 == null_ptr);
  ASIO_CHECK(ex_no_props_4 == ex_no_props_1);

  ex_no_props_t ex_no_props_5(std::nothrow, ex_no_props_3);

  ASIO_CHECK(ex_no_props_5.target<void>() != 0);
  ASIO_CHECK(ex_no_props_5 != null_ptr);
  ASIO_CHECK(ex_no_props_5 == ex_no_props_3);

  ex_no_props_t ex_no_props_6(std::nothrow, fat_executor(1));

  ASIO_CHECK(ex_no_props_6.target<void>() != 0);
  ASIO_CHECK(ex_no_props_6 != null_ptr);
  ASIO_CHECK(ex_no_props_6 != ex_no_props_1);

  ex_no_props_t ex_no_props_7(std::nothrow, fat_executor(1));

  ASIO_CHECK(ex_no_props_7.target<void>() != 0);
  ASIO_CHECK(ex_no_props_7 != null_ptr);
  ASIO_CHECK(ex_no_props_7 != ex_no_props_1);
  ASIO_CHECK(ex_no_props_7 == ex_no_props_6);

  ex_no_props_t ex_no_props_8(std::nothrow, fat_executor(2));

  ASIO_CHECK(ex_no_props_8.target<void>() != 0);
  ASIO_CHECK(ex_no_props_8 != null_ptr);
  ASIO_CHECK(ex_no_props_8 != ex_no_props_1);
  ASIO_CHECK(ex_no_props_8 != ex_no_props_6);
  ASIO_CHECK(ex_no_props_8 != ex_no_props_7);

  ex_no_props_t ex_no_props_9(std::nothrow, ex_no_props_6);

  ASIO_CHECK(ex_no_props_9.target<void>() != 0);
  ASIO_CHECK(ex_no_props_9 != null_ptr);
  ASIO_CHECK(ex_no_props_9 != ex_no_props_1);
  ASIO_CHECK(ex_no_props_9 == ex_no_props_6);
  ASIO_CHECK(ex_no_props_9 == ex_no_props_7);
  ASIO_CHECK(ex_no_props_9 != ex_no_props_8);

  ex_no_props_t ex_no_props_10(std::nothrow, std::move(ex_no_props_1));

  ASIO_CHECK(ex_no_props_10.target<void>() == 0);
  ASIO_CHECK(ex_no_props_10 == null_ptr);
  ASIO_CHECK(ex_no_props_1.target<void>() == 0);
  ASIO_CHECK(ex_no_props_1 == null_ptr);

  ex_no_props_t ex_no_props_11(std::nothrow, std::move(ex_no_props_3));

  ASIO_CHECK(ex_no_props_11.target<void>() != 0);
  ASIO_CHECK(ex_no_props_11 != null_ptr);
  ASIO_CHECK(ex_no_props_3.target<void>() == 0);
  ASIO_CHECK(ex_no_props_3 == null_ptr);
  ASIO_CHECK(ex_no_props_11 == ex_no_props_5);

  ex_no_props_t ex_no_props_12(std::nothrow, std::move(ex_no_props_7));

  ASIO_CHECK(ex_no_props_12.target<void>() != 0);
  ASIO_CHECK(ex_no_props_12 != null_ptr);
  ASIO_CHECK(ex_no_props_7.target<void>() == 0);
  ASIO_CHECK(ex_no_props_7 == null_ptr);
  ASIO_CHECK(ex_no_props_12 == ex_no_props_6);
  ASIO_CHECK(ex_no_props_12 != ex_no_props_8);

  ex_no_props_t ex_no_props_13(std::nothrow, ex_two_props_1);

  ASIO_CHECK(ex_no_props_13.target<void>() == 0);
  ASIO_CHECK(ex_no_props_13 == null_ptr);

  ex_no_props_t ex_no_props_14(std::nothrow, ex_two_props_5);

  ASIO_CHECK(ex_no_props_14.target<void>() != 0);
  ASIO_CHECK(ex_no_props_14 != null_ptr);

  ex_no_props_t ex_no_props_15(std::nothrow, ex_two_props_9);

  ASIO_CHECK(ex_no_props_15.target<void>() != 0);
  ASIO_CHECK(ex_no_props_15 != null_ptr);

  ex_no_props_t ex_no_props_16(std::nothrow, ex_one_prop_1);

  ASIO_CHECK(ex_no_props_16.target<void>() == 0);
  ASIO_CHECK(ex_no_props_16 == null_ptr);

  ex_no_props_t ex_no_props_17(std::nothrow, ex_one_prop_5);

  ASIO_CHECK(ex_no_props_17.target<void>() != 0);
  ASIO_CHECK(ex_no_props_17 != null_ptr);

  ex_no_props_t ex_no_props_18(std::nothrow, ex_one_prop_9);

  ASIO_CHECK(ex_no_props_18.target<void>() != 0);
  ASIO_CHECK(ex_no_props_18 != null_ptr);

  next_nothrow_new_fails = true;
  ex_no_props_t ex_no_props_19(std::nothrow, fat_executor(3));

  ASIO_CHECK(ex_no_props_19.target<void>() == 0);
  ASIO_CHECK(ex_no_props_19 == null_ptr);
  ASIO_CHECK(ex_no_props_19 == ex_no_props_1);
}

void any_executor_assignment_test()
{
  typedef execution::any_executor<> ex_no_props_t;

  typedef execution::any_executor<
      execution::blocking_t
    > ex_one_prop_t;

  typedef execution::any_executor<
      execution::blocking_t,
      execution::occupancy_t
    > ex_two_props_t;

  thread_pool pool(1);
  asio::nullptr_t null_ptr = asio::nullptr_t();

  ex_two_props_t ex_two_props_1;

  ex_two_props_t ex_two_props_2;
  ex_two_props_2 = null_ptr;

  ASIO_CHECK(ex_two_props_2.target<void>() == 0);

  ex_two_props_t ex_two_props_3;
  ex_two_props_3 = pool.executor();

  ASIO_CHECK(ex_two_props_3.target<void>() != 0);

  ex_two_props_t ex_two_props_4;
  ex_two_props_4 = ex_two_props_1;

  ASIO_CHECK(ex_two_props_4.target<void>() == 0);
  ASIO_CHECK(ex_two_props_4 == ex_two_props_1);

  ex_two_props_4 = ex_two_props_3;

  ASIO_CHECK(ex_two_props_4.target<void>() != 0);
  ASIO_CHECK(ex_two_props_4 == ex_two_props_3);

  ex_two_props_t ex_two_props_5;
  ex_two_props_5 = fat_executor(1);

  ASIO_CHECK(ex_two_props_5.target<void>() != 0);
  ASIO_CHECK(ex_two_props_5 != null_ptr);
  ASIO_CHECK(ex_two_props_5 != ex_two_props_1);

  ex_two_props_t ex_two_props_6;
  ex_two_props_6 = fat_executor(1);

  ASIO_CHECK(ex_two_props_6.target<void>() != 0);
  ASIO_CHECK(ex_two_props_6 != null_ptr);
  ASIO_CHECK(ex_two_props_6 != ex_two_props_1);
  ASIO_CHECK(ex_two_props_6 == ex_two_props_5);

  ex_two_props_6 = fat_executor(2);

  ASIO_CHECK(ex_two_props_6.target<void>() != 0);
  ASIO_CHECK(ex_two_props_6 != null_ptr);
  ASIO_CHECK(ex_two_props_6 != ex_two_props_1);
  ASIO_CHECK(ex_two_props_6 != ex_two_props_5);

  ex_two_props_t ex_two_props_7;
  ex_two_props_7 = ex_two_props_5;

  ASIO_CHECK(ex_two_props_7.target<void>() != 0);
  ASIO_CHECK(ex_two_props_7 != null_ptr);
  ASIO_CHECK(ex_two_props_7 != ex_two_props_1);
  ASIO_CHECK(ex_two_props_7 == ex_two_props_5);
  ASIO_CHECK(ex_two_props_7 != ex_two_props_6);

  ex_two_props_t ex_two_props_8;
  ex_two_props_8 = std::move(ex_two_props_1);

  ASIO_CHECK(ex_two_props_8.target<void>() == 0);
  ASIO_CHECK(ex_two_props_1.target<void>() == 0);

  ex_two_props_8 = std::move(ex_two_props_3);

  ASIO_CHECK(ex_two_props_8.target<void>() != 0);
  ASIO_CHECK(ex_two_props_3.target<void>() == 0);
  ASIO_CHECK(ex_two_props_8 == ex_two_props_4);

  ex_two_props_8 = std::move(ex_two_props_5);

  ASIO_CHECK(ex_two_props_8.target<void>() != 0);
  ASIO_CHECK(ex_two_props_5.target<void>() == 0);
  ASIO_CHECK(ex_two_props_8 == ex_two_props_7);

  ex_one_prop_t ex_one_prop_1;

  ex_one_prop_t ex_one_prop_2;
  ex_one_prop_2 = null_ptr;

  ASIO_CHECK(ex_one_prop_2.target<void>() == 0);

  ex_one_prop_t ex_one_prop_3;
  ex_one_prop_3 = pool.executor();

  ASIO_CHECK(ex_one_prop_3.target<void>() != 0);

  ex_one_prop_t ex_one_prop_4;
  ex_one_prop_4 = ex_one_prop_1;

  ASIO_CHECK(ex_one_prop_4.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_4 == ex_one_prop_1);

  ex_one_prop_4 = ex_one_prop_3;

  ASIO_CHECK(ex_one_prop_4.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_4 == ex_one_prop_3);

  ex_one_prop_t ex_one_prop_5;
  ex_one_prop_5 = fat_executor(1);

  ASIO_CHECK(ex_one_prop_5.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_5 != null_ptr);
  ASIO_CHECK(ex_one_prop_5 != ex_one_prop_1);

  ex_one_prop_t ex_one_prop_6;
  ex_one_prop_6 = fat_executor(1);

  ASIO_CHECK(ex_one_prop_6.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_6 != null_ptr);
  ASIO_CHECK(ex_one_prop_6 != ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_6 == ex_one_prop_5);

  ex_one_prop_6 = fat_executor(2);

  ASIO_CHECK(ex_one_prop_6.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_6 != null_ptr);
  ASIO_CHECK(ex_one_prop_6 != ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_6 != ex_one_prop_5);

  ex_one_prop_t ex_one_prop_7;
  ex_one_prop_7 = ex_one_prop_5;

  ASIO_CHECK(ex_one_prop_7.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_7 != null_ptr);
  ASIO_CHECK(ex_one_prop_7 != ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_7 == ex_one_prop_5);
  ASIO_CHECK(ex_one_prop_7 != ex_one_prop_6);

  ex_one_prop_t ex_one_prop_8;
  ex_one_prop_8 = std::move(ex_one_prop_1);

  ASIO_CHECK(ex_one_prop_8.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_1.target<void>() == 0);

  ex_one_prop_8 = std::move(ex_one_prop_3);

  ASIO_CHECK(ex_one_prop_8.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_3.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_8 == ex_one_prop_4);

  ex_one_prop_8 = std::move(ex_one_prop_5);

  ASIO_CHECK(ex_one_prop_8.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_5.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_8 == ex_one_prop_7);

  ex_one_prop_t ex_one_prop_9;
  ex_one_prop_9 = ex_two_props_1;

  ASIO_CHECK(ex_one_prop_9.target<void>() == 0);

  ex_one_prop_9 = ex_two_props_4;

  ASIO_CHECK(ex_one_prop_9.target<void>() != 0);

  ex_one_prop_9 = ex_two_props_7;

  ASIO_CHECK(ex_one_prop_9.target<void>() != 0);

  ex_no_props_t ex_no_props_1;

  ex_no_props_t ex_no_props_2;
  ex_no_props_2 = null_ptr;

  ASIO_CHECK(ex_no_props_2.target<void>() == 0);

  ex_no_props_t ex_no_props_3;
  ex_no_props_3 = pool.executor();

  ASIO_CHECK(ex_no_props_3.target<void>() != 0);

  ex_no_props_t ex_no_props_4;
  ex_no_props_4 = ex_no_props_1;

  ASIO_CHECK(ex_no_props_4.target<void>() == 0);
  ASIO_CHECK(ex_no_props_4 == ex_no_props_1);

  ex_no_props_4 = ex_no_props_3;

  ASIO_CHECK(ex_no_props_4.target<void>() != 0);
  ASIO_CHECK(ex_no_props_4 == ex_no_props_3);

  ex_no_props_t ex_no_props_5;
  ex_no_props_5 = fat_executor(1);

  ASIO_CHECK(ex_no_props_5.target<void>() != 0);
  ASIO_CHECK(ex_no_props_5 != null_ptr);
  ASIO_CHECK(ex_no_props_5 != ex_no_props_1);

  ex_no_props_t ex_no_props_6;
  ex_no_props_6 = fat_executor(1);

  ASIO_CHECK(ex_no_props_6.target<void>() != 0);
  ASIO_CHECK(ex_no_props_6 != null_ptr);
  ASIO_CHECK(ex_no_props_6 != ex_no_props_1);
  ASIO_CHECK(ex_no_props_6 == ex_no_props_5);

  ex_no_props_6 = fat_executor(2);

  ASIO_CHECK(ex_no_props_6.target<void>() != 0);
  ASIO_CHECK(ex_no_props_6 != null_ptr);
  ASIO_CHECK(ex_no_props_6 != ex_no_props_1);
  ASIO_CHECK(ex_no_props_6 != ex_no_props_5);

  ex_no_props_t ex_no_props_7;
  ex_no_props_7 = ex_no_props_5;

  ASIO_CHECK(ex_no_props_7.target<void>() != 0);
  ASIO_CHECK(ex_no_props_7 != null_ptr);
  ASIO_CHECK(ex_no_props_7 != ex_no_props_1);
  ASIO_CHECK(ex_no_props_7 == ex_no_props_5);
  ASIO_CHECK(ex_no_props_7 != ex_no_props_6);

  ex_no_props_t ex_no_props_8;
  ex_no_props_8 = std::move(ex_no_props_1);

  ASIO_CHECK(ex_no_props_8.target<void>() == 0);
  ASIO_CHECK(ex_no_props_1.target<void>() == 0);

  ex_no_props_8 = std::move(ex_no_props_3);

  ASIO_CHECK(ex_no_props_8.target<void>() != 0);
  ASIO_CHECK(ex_no_props_3.target<void>() == 0);
  ASIO_CHECK(ex_no_props_8 == ex_no_props_4);

  ex_no_props_8 = std::move(ex_no_props_5);

  ASIO_CHECK(ex_no_props_8.target<void>() != 0);
  ASIO_CHECK(ex_no_props_5.target<void>() == 0);
  ASIO_CHECK(ex_no_props_8 == ex_no_props_7);

  ex_no_props_t ex_no_props_9;
  ex_no_props_9 = ex_two_props_1;

  ASIO_CHECK(ex_no_props_9.target<void>() == 0);

  ex_no_props_9 = ex_two_props_4;

  ASIO_CHECK(ex_no_props_9.target<void>() != 0);

  ex_no_props_9 = ex_two_props_7;

  ASIO_CHECK(ex_no_props_9.target<void>() != 0);

  ex_no_props_9 = ex_one_prop_1;

  ASIO_CHECK(ex_no_props_9.target<void>() == 0);

  ex_no_props_9 = ex_one_prop_4;

  ASIO_CHECK(ex_no_props_9.target<void>() != 0);

  ex_no_props_9 = ex_one_prop_7;

  ASIO_CHECK(ex_no_props_9.target<void>() != 0);
}

void any_executor_swap_test()
{
  typedef execution::any_executor<> ex_no_props_t;

  typedef execution::any_executor<
      execution::blocking_t
    > ex_one_prop_t;

  typedef execution::any_executor<
      execution::blocking_t,
      execution::occupancy_t
    > ex_two_props_t;

  thread_pool pool1(1);
  thread_pool pool2(1);

  ex_no_props_t ex_no_props_1(pool1.executor());
  ex_no_props_t ex_no_props_2(pool2.executor());

  ex_no_props_t ex_no_props_3(ex_no_props_1);
  ex_no_props_t ex_no_props_4(ex_no_props_2);

  ASIO_CHECK(ex_no_props_3 == ex_no_props_1);
  ASIO_CHECK(ex_no_props_4 == ex_no_props_2);

  ex_no_props_3.swap(ex_no_props_4);

  ASIO_CHECK(ex_no_props_3 == ex_no_props_2);
  ASIO_CHECK(ex_no_props_4 == ex_no_props_1);

  execution::swap(ex_no_props_3, ex_no_props_4);

  ASIO_CHECK(ex_no_props_3 == ex_no_props_1);
  ASIO_CHECK(ex_no_props_4 == ex_no_props_2);

  ex_one_prop_t ex_one_prop_1(pool1.executor());
  ex_one_prop_t ex_one_prop_2(pool2.executor());

  ex_one_prop_t ex_one_prop_3(ex_one_prop_1);
  ex_one_prop_t ex_one_prop_4(ex_one_prop_2);

  ASIO_CHECK(ex_one_prop_3 == ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_4 == ex_one_prop_2);

  ex_one_prop_3.swap(ex_one_prop_4);

  ASIO_CHECK(ex_one_prop_3 == ex_one_prop_2);
  ASIO_CHECK(ex_one_prop_4 == ex_one_prop_1);

  execution::swap(ex_one_prop_3, ex_one_prop_4);

  ASIO_CHECK(ex_one_prop_3 == ex_one_prop_1);
  ASIO_CHECK(ex_one_prop_4 == ex_one_prop_2);

  ex_two_props_t ex_two_props_1(pool1.executor());
  ex_two_props_t ex_two_props_2(pool2.executor());

  ex_two_props_t ex_two_props_3(ex_two_props_1);
  ex_two_props_t ex_two_props_4(ex_two_props_2);

  ASIO_CHECK(ex_two_props_3 == ex_two_props_1);
  ASIO_CHECK(ex_two_props_4 == ex_two_props_2);

  ex_two_props_3.swap(ex_two_props_4);

  ASIO_CHECK(ex_two_props_3 == ex_two_props_2);
  ASIO_CHECK(ex_two_props_4 == ex_two_props_1);

  execution::swap(ex_two_props_3, ex_two_props_4);

  ASIO_CHECK(ex_two_props_3 == ex_two_props_1);
  ASIO_CHECK(ex_two_props_4 == ex_two_props_2);
}

void any_executor_query_test()
{
  thread_pool pool(1);
  execution::any_executor<
      execution::blocking_t,
      execution::outstanding_work_t,
      execution::relationship_t,
      execution::mapping_t::thread_t,
      execution::occupancy_t>
    ex(pool.executor());

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

  ASIO_CHECK(
      asio::query(ex, asio::execution::mapping)
        == asio::execution::mapping.thread);

  ASIO_CHECK(
      asio::query(ex, asio::execution::occupancy)
        == 1);
}

void any_executor_execute_test()
{
  int count = 0;
  thread_pool pool(1);
  execution::any_executor<
      execution::blocking_t::possibly_t,
      execution::blocking_t::never_t,
      execution::outstanding_work_t::untracked_t,
      execution::outstanding_work_t::tracked_t,
      execution::relationship_t::continuation_t>
    ex(pool.executor());

  ex.execute(bindns::bind(increment, &count));

  asio::require(ex, asio::execution::blocking.possibly).execute(
      bindns::bind(increment, &count));

  asio::require(ex, asio::execution::blocking.never).execute(
      bindns::bind(increment, &count));

  asio::require(ex,
      asio::execution::blocking.never,
      asio::execution::outstanding_work.tracked
    ).execute(bindns::bind(increment, &count));

  asio::require(ex,
      asio::execution::blocking.never,
      asio::execution::outstanding_work.untracked
    ).execute(bindns::bind(increment, &count));

  asio::require(ex,
      asio::execution::blocking.never,
      asio::execution::outstanding_work.untracked,
      asio::execution::relationship.continuation
    ).execute(bindns::bind(increment, &count));

  pool.wait();

  ASIO_CHECK(count == 6);
}

// An operation that records how it was run. A target that takes a ready-made
// operation as it stands runs it from the scheduler, which passes itself as
// the owner. One that wraps it in a second operation runs the inner operation
// from that wrapper, which passes the operation itself as the owner.
struct recording_operation : asio::detail::operation
{
  int* count_;
  bool* wrapped_;

  recording_operation(int* count, bool* wrapped)
    : asio::detail::operation(&recording_operation::do_complete),
      count_(count),
      wrapped_(wrapped)
  {
  }

  static void do_complete(void* owner, asio::detail::operation* base,
      int /*error_value*/, std::size_t /*bytes_transferred*/)
  {
    recording_operation* o = static_cast<recording_operation*>(base);
    if (owner)
    {
      ++*o->count_;
      if (owner == static_cast<void*>(o))
        *o->wrapped_ = true;
    }
    delete o;
  }
};

// The overload set that any_executor::execute presents, reproduced here with
// distinguishable return types so that the compiler's choice between the two
// candidates can be observed directly rather than assumed.
struct chose_template
{
};

struct chose_ready_made
{
};

struct execute_overload_probe
{
  template <typename F>
  chose_template execute(F&&) const;

  chose_ready_made execute(asio::detail::executor_function&&) const;
};

template <typename Arg>
struct execute_overload_chosen
{
  typedef decltype(
      std::declval<const execute_overload_probe&>().execute(
        std::declval<Arg>())) type;
};

typedef asio::detail::executor_function ready_made;

// A ready-made operation selects the non-template overload.
static_assert(std::is_same<
    execute_overload_chosen<ready_made>::type, chose_ready_made>::value,
    "an rvalue executor_function selects the non-template overload");

// Everything else, an lvalue executor_function included, is unaffected.
static_assert(std::is_same<
    execute_overload_chosen<ready_made&>::type, chose_template>::value,
    "an lvalue executor_function still selects the template");

static_assert(std::is_same<
    execute_overload_chosen<const ready_made&>::type, chose_template>::value,
    "a const lvalue executor_function still selects the template");

static_assert(std::is_same<
    execute_overload_chosen<const ready_made>::type, chose_template>::value,
    "a const rvalue executor_function still selects the template");

static_assert(std::is_same<
    execute_overload_chosen<void (*)()>::type, chose_template>::value,
    "a plain function pointer still selects the template");

static_assert(std::is_same<
    execute_overload_chosen<int>::type, chose_template>::value,
    "an unrelated argument still selects the template");

void any_executor_execute_ready_made_test()
{
  int count = 0;

  asio::io_context ioc(1);
  asio::any_io_executor ex(ioc.get_executor());

  // A ready-made operation submitted to a target that queues it.
  ex.execute(asio::detail::executor_function(
        bindns::bind(increment, &count), std::allocator<void>()));

  // And to a target reached by way of a strand.
  asio::strand<asio::any_io_executor> st(ex);
  asio::any_io_executor stex(st);
  stex.execute(asio::detail::executor_function(
        bindns::bind(increment, &count), std::allocator<void>()));

  ioc.run();

  ASIO_CHECK(count == 2);

  // An always-blocking target runs the operation in place.
  thread_pool pool(1);
  execution::any_executor<
      execution::blocking_t::possibly_t,
      execution::blocking_t::always_t,
      execution::blocking_t::never_t>
    bex(asio::require(pool.executor(), asio::execution::blocking.always));

  bex.execute(asio::detail::executor_function(
        bindns::bind(increment, &count), std::allocator<void>()));

  ASIO_CHECK(count == 3);

  // An executor with no target throws, and the operation is destroyed with
  // the function it holds rather than run.
  asio::any_io_executor empty;
  bool caught = false;
  try
  {
    empty.execute(asio::detail::executor_function(
          bindns::bind(increment, &count), std::allocator<void>()));
  }
  catch (asio::execution::bad_executor&)
  {
    caught = true;
  }

  ASIO_CHECK(caught);
  ASIO_CHECK(count == 3);

  pool.wait();

  // The operation is handed to the target as it stands. Were it wrapped in a
  // second operation instead, the inner one would be run by that wrapper
  // rather than by the scheduler.
  int handed_count = 0;
  bool handed_wrapped = false;
  ioc.restart();
  ex.execute(asio::detail::executor_function(
        new recording_operation(&handed_count, &handed_wrapped)));
  ioc.run();

  ASIO_CHECK(handed_count == 1);
  ASIO_CHECK(!handed_wrapped);
}

void executor_targets_equal_test()
{
  asio::io_context ioc1;
  asio::io_context ioc2;
  asio::any_io_executor poly(ioc1.get_executor());

  // A polymorphic executor against the concrete executor it erases,
  // either way round.
  ASIO_CHECK(asio::execution::detail::executor_targets_equal(
        poly, ioc1.get_executor()));
  ASIO_CHECK(asio::execution::detail::executor_targets_equal(
        ioc1.get_executor(), poly));

  // The same concrete type, a different executor: must NOT be equal, or a
  // completion would be run on the wrong context's behalf.
  ASIO_CHECK(!asio::execution::detail::executor_targets_equal(
        poly, ioc2.get_executor()));
  ASIO_CHECK(!asio::execution::detail::executor_targets_equal(
        ioc2.get_executor(), poly));

  // Two polymorphic executors of differing any_executor<...> instantiations:
  // compared on the any_executor_base layer they share, since a direct ==
  // between them would be ambiguous. The same target, either way round.
  asio::any_completion_executor completion_poly(ioc1.get_executor());
  ASIO_CHECK(asio::execution::detail::executor_targets_equal(
        poly, completion_poly));
  ASIO_CHECK(asio::execution::detail::executor_targets_equal(
        completion_poly, poly));

  // The same pair over differing contexts: must NOT be equal.
  asio::any_completion_executor completion_poly2(ioc2.get_executor());
  ASIO_CHECK(!asio::execution::detail::executor_targets_equal(
        poly, completion_poly2));
  ASIO_CHECK(!asio::execution::detail::executor_targets_equal(
        completion_poly2, poly));

  // A different concrete type altogether.
  asio::thread_pool pool(1);
  ASIO_CHECK(!asio::execution::detail::executor_targets_equal(
        poly, pool.get_executor()));

  // An executor with no target matches nothing.
  asio::any_io_executor empty;
  ASIO_CHECK(!asio::execution::detail::executor_targets_equal(
        empty, ioc1.get_executor()));

  // A target that carries tracked outstanding work is a distinct target,
  // and so does not match the plain executor it was preferred from.
  asio::any_io_executor tracked(
      asio::prefer(poly, asio::execution::outstanding_work.tracked));
  ASIO_CHECK(!asio::execution::detail::executor_targets_equal(
        tracked, ioc1.get_executor()));

  // Two differing concrete types are never equal.
  ASIO_CHECK(!asio::execution::detail::executor_targets_equal(
        ioc1.get_executor(), pool.get_executor()));
}

ASIO_TEST_SUITE
(
  "any_executor",
  ASIO_TEST_CASE(executor_targets_equal_test)
  ASIO_TEST_CASE(any_executor_construction_test)
  ASIO_TEST_CASE(any_executor_nothrow_construction_test)
  ASIO_TEST_CASE(any_executor_assignment_test)
  ASIO_TEST_CASE(any_executor_swap_test)
  ASIO_TEST_CASE(any_executor_query_test)
  ASIO_TEST_CASE(any_executor_execute_test)
  ASIO_TEST_CASE(any_executor_execute_ready_made_test)
)
