//
// any_executor.cpp
// ~~~~~~~~~~~~~~~~
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
#include "asio/execution/any_executor.hpp"

#include "asio/thread_pool.hpp"
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

  ex_two_props_t ex_two_props_2(null_ptr);

  ASIO_CHECK(ex_two_props_2.target<void>() == 0);
  ASIO_CHECK(ex_two_props_2 == ex_two_props_1);

  ex_two_props_t ex_two_props_3(pool.executor());

  ASIO_CHECK(ex_two_props_3.target<void>() != 0);
  ASIO_CHECK(ex_two_props_3 != ex_two_props_1);

  ex_two_props_t ex_two_props_4(ex_two_props_1);

  ASIO_CHECK(ex_two_props_4.target<void>() == 0);
  ASIO_CHECK(ex_two_props_4 == ex_two_props_1);

  ex_two_props_t ex_two_props_5(ex_two_props_3);

  ASIO_CHECK(ex_two_props_5.target<void>() != 0);
  ASIO_CHECK(ex_two_props_5 == ex_two_props_3);

#if defined(ASIO_HAS_MOVE)
  ex_two_props_t ex_two_props_6(std::move(ex_two_props_1));

  ASIO_CHECK(ex_two_props_6.target<void>() == 0);
  ASIO_CHECK(ex_two_props_1.target<void>() == 0);

  ex_two_props_t ex_two_props_7(std::move(ex_two_props_3));

  ASIO_CHECK(ex_two_props_7.target<void>() != 0);
  ASIO_CHECK(ex_two_props_3.target<void>() == 0);
  ASIO_CHECK(ex_two_props_7 == ex_two_props_5);
#endif // defined(ASIO_HAS_MOVE)

  ex_one_prop_t ex_one_prop_1;

  ASIO_CHECK(ex_one_prop_1.target<void>() == 0);

  ex_one_prop_t ex_one_prop_2(null_ptr);

  ASIO_CHECK(ex_one_prop_2.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_2 == ex_one_prop_1);

  ex_one_prop_t ex_one_prop_3(pool.executor());

  ASIO_CHECK(ex_one_prop_3.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_3 != ex_one_prop_1);

  ex_one_prop_t ex_one_prop_4(ex_one_prop_1);

  ASIO_CHECK(ex_one_prop_4.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_4 == ex_one_prop_1);

  ex_one_prop_t ex_one_prop_5(ex_one_prop_3);

  ASIO_CHECK(ex_one_prop_5.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_5 == ex_one_prop_3);

#if defined(ASIO_HAS_MOVE)
  ex_one_prop_t ex_one_prop_6(std::move(ex_one_prop_1));

  ASIO_CHECK(ex_one_prop_6.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_1.target<void>() == 0);

  ex_one_prop_t ex_one_prop_7(std::move(ex_one_prop_3));

  ASIO_CHECK(ex_one_prop_7.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_3.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_7 == ex_one_prop_5);
#endif // defined(ASIO_HAS_MOVE)

  ex_one_prop_t ex_one_prop_8(ex_two_props_1);

  ASIO_CHECK(ex_one_prop_8.target<void>() == 0);

  ex_one_prop_t ex_one_prop_9(ex_two_props_5);

  ASIO_CHECK(ex_one_prop_9.target<void>() != 0);

  ex_no_props_t ex_no_props_1;

  ASIO_CHECK(ex_no_props_1.target<void>() == 0);

  ex_no_props_t ex_no_props_2(null_ptr);

  ASIO_CHECK(ex_no_props_2.target<void>() == 0);
  ASIO_CHECK(ex_no_props_2 == ex_no_props_1);

  ex_no_props_t ex_no_props_3(pool.executor());

  ASIO_CHECK(ex_no_props_3.target<void>() != 0);
  ASIO_CHECK(ex_no_props_3 != ex_no_props_1);

  ex_no_props_t ex_no_props_4(ex_no_props_1);

  ASIO_CHECK(ex_no_props_4.target<void>() == 0);
  ASIO_CHECK(ex_no_props_4 == ex_no_props_1);

  ex_no_props_t ex_no_props_5(ex_no_props_3);

  ASIO_CHECK(ex_no_props_5.target<void>() != 0);
  ASIO_CHECK(ex_no_props_5 == ex_no_props_3);

#if defined(ASIO_HAS_MOVE)
  ex_no_props_t ex_no_props_6(std::move(ex_no_props_1));

  ASIO_CHECK(ex_no_props_6.target<void>() == 0);
  ASIO_CHECK(ex_no_props_1.target<void>() == 0);

  ex_no_props_t ex_no_props_7(std::move(ex_no_props_3));

  ASIO_CHECK(ex_no_props_7.target<void>() != 0);
  ASIO_CHECK(ex_no_props_3.target<void>() == 0);
  ASIO_CHECK(ex_no_props_7 == ex_no_props_5);
#endif // defined(ASIO_HAS_MOVE)

  ex_no_props_t ex_no_props_8(ex_two_props_1);

  ASIO_CHECK(ex_no_props_8.target<void>() == 0);

  ex_no_props_t ex_no_props_9(ex_two_props_5);

  ASIO_CHECK(ex_no_props_9.target<void>() != 0);

  ex_no_props_t ex_no_props_10(ex_one_prop_1);

  ASIO_CHECK(ex_no_props_10.target<void>() == 0);

  ex_no_props_t ex_no_props_11(ex_one_prop_5);

  ASIO_CHECK(ex_no_props_11.target<void>() != 0);
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

#if defined(ASIO_HAS_MOVE)
  ex_two_props_t ex_two_props_5;
  ex_two_props_5 = std::move(ex_two_props_1);

  ASIO_CHECK(ex_two_props_5.target<void>() == 0);
  ASIO_CHECK(ex_two_props_1.target<void>() == 0);

  ex_two_props_5 = std::move(ex_two_props_3);

  ASIO_CHECK(ex_two_props_5.target<void>() != 0);
  ASIO_CHECK(ex_two_props_3.target<void>() == 0);
  ASIO_CHECK(ex_two_props_5 == ex_two_props_4);
#endif // defined(ASIO_HAS_MOVE)

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

#if defined(ASIO_HAS_MOVE)
  ex_one_prop_t ex_one_prop_5;
  ex_one_prop_5 = std::move(ex_one_prop_1);

  ASIO_CHECK(ex_one_prop_5.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_1.target<void>() == 0);

  ex_one_prop_5 = std::move(ex_one_prop_3);

  ASIO_CHECK(ex_one_prop_5.target<void>() != 0);
  ASIO_CHECK(ex_one_prop_3.target<void>() == 0);
  ASIO_CHECK(ex_one_prop_5 == ex_one_prop_4);
#endif // defined(ASIO_HAS_MOVE)

  ex_one_prop_t ex_one_prop_6;
  ex_one_prop_6 = ex_two_props_1;

  ASIO_CHECK(ex_one_prop_6.target<void>() == 0);

  ex_one_prop_6 = ex_two_props_4;

  ASIO_CHECK(ex_one_prop_6.target<void>() != 0);

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

#if defined(ASIO_HAS_MOVE)
  ex_no_props_t ex_no_props_5;
  ex_no_props_5 = std::move(ex_no_props_1);

  ASIO_CHECK(ex_no_props_5.target<void>() == 0);
  ASIO_CHECK(ex_no_props_1.target<void>() == 0);

  ex_no_props_5 = std::move(ex_no_props_3);

  ASIO_CHECK(ex_no_props_5.target<void>() != 0);
  ASIO_CHECK(ex_no_props_3.target<void>() == 0);
  ASIO_CHECK(ex_no_props_5 == ex_no_props_4);
#endif // defined(ASIO_HAS_MOVE)

  ex_no_props_t ex_no_props_6;
  ex_no_props_6 = ex_two_props_1;

  ASIO_CHECK(ex_no_props_6.target<void>() == 0);

  ex_no_props_6 = ex_two_props_4;

  ASIO_CHECK(ex_no_props_6.target<void>() != 0);

  ex_no_props_6 = ex_one_prop_1;

  ASIO_CHECK(ex_no_props_6.target<void>() == 0);

  ex_no_props_6 = ex_one_prop_4;

  ASIO_CHECK(ex_no_props_6.target<void>() != 0);
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

  asio::execution::execute(pool.executor(),
      bindns::bind(increment, &count));

  asio::execution::execute(
      asio::require(pool.executor(),
        asio::execution::blocking.possibly),
      bindns::bind(increment, &count));

  asio::execution::execute(
      asio::require(pool.executor(),
        asio::execution::blocking.never),
      bindns::bind(increment, &count));

  asio::execution::execute(
      asio::require(pool.executor(),
        asio::execution::blocking.never,
        asio::execution::outstanding_work.tracked),
      bindns::bind(increment, &count));

  asio::execution::execute(
      asio::require(pool.executor(),
        asio::execution::blocking.never,
        asio::execution::outstanding_work.untracked),
      bindns::bind(increment, &count));

  asio::execution::execute(
      asio::require(pool.executor(),
        asio::execution::blocking.never,
        asio::execution::outstanding_work.untracked,
        asio::execution::relationship.continuation),
      bindns::bind(increment, &count));

  pool.wait();

  ASIO_CHECK(count == 6);
}

ASIO_TEST_SUITE
(
  "any_executor",
  ASIO_TEST_CASE(any_executor_construction_test)
  ASIO_TEST_CASE(any_executor_assignment_test)
  ASIO_TEST_CASE(any_executor_query_test)
  ASIO_TEST_CASE(any_executor_execute_test)
)
