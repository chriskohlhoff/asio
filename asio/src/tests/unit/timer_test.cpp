//
// timer_test.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <boost/bind.hpp>
#include "asio.hpp"
#include "unit_test.hpp"

void increment(int* count)
{
  ++(*count);
}

void decrement_to_zero(asio::timer* t, int* count)
{
  if (*count > 0)
  {
    --(*count);

    int before_value = *count;

    t->expiry(t->expiry() + 1);
    t->async_wait(boost::bind(decrement_to_zero, t, count));

    // Completion cannot nest, so count value should remain unchanged.
    UNIT_TEST_CHECK(*count == before_value);
  }
}

void increment_if_not_cancelled(int* count, const asio::error& e)
{
  if (!e)
    ++(*count);
}

void cancel_timer(asio::timer* t)
{
  int num_cancelled = t->cancel();
  UNIT_TEST_CHECK(num_cancelled == 1);
}

void timer_test()
{
  asio::demuxer d;
  int count = 0;

  asio::time start = asio::time::now();

  asio::timer t1(d, asio::time::now() + 1);
  t1.wait();

  // The timer must block until after its expiry time.
  asio::time end = asio::time::now();
  asio::time expected_end = start + 1;
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  start = asio::time::now();

  asio::timer t2(d, asio::time::now() + asio::time(1, 500000));
  t2.wait();

  // The timer must block until after its expiry time.
  end = asio::time::now();
  expected_end = start + asio::time(1, 500000);
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  t2.expiry(t2.expiry() + 1);
  t2.wait();

  // The timer must block until after its expiry time.
  end = asio::time::now();
  expected_end += 1;
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  t2.expiry(t2.expiry() + asio::time(1, 200000));
  t2.wait();

  // The timer must block until after its expiry time.
  end = asio::time::now();
  expected_end += asio::time(1, 200000);
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  start = asio::time::now();

  asio::timer t3(d, asio::time::now() + 1);
  t3.async_wait(boost::bind(increment, &count));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all operations have finished, and
  // this should not be until after the timer's expiry time.
  UNIT_TEST_CHECK(count == 1);
  end = asio::time::now();
  expected_end = start + 1;
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  count = 3;
  start = asio::time::now();

  asio::timer t4(d, asio::time::now() + 1);
  t4.async_wait(boost::bind(decrement_to_zero, &t4, &count));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 3);

  d.reset();
  d.run();

  // The run() call will not return until all operations have finished, and
  // this should not be until after the timer's final expiry time.
  UNIT_TEST_CHECK(count == 0);
  end = asio::time::now();
  expected_end = start + 3;
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  count = 0;
  start = asio::time::now();

  asio::timer t5(d, asio::time::now() + 10);
  t5.async_wait(boost::bind(increment_if_not_cancelled, &count,
        asio::arg::error));
  asio::timer t6(d, asio::time::now() + 1);
  t6.async_wait(boost::bind(cancel_timer, &t5));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.reset();
  d.run();

  // The timer should have been cancelled, so count should not have changed.
  // The total run time should not have been much more than 1 second (and
  // certainly far less than 10 seconds).
  UNIT_TEST_CHECK(count == 0);
  end = asio::time::now();
  expected_end = start + 2;
  UNIT_TEST_CHECK(end < expected_end);

  // Wait on the timer again without cancelling it. This time the asynchronous
  // wait should run to completion and increment the counter.
  t5.async_wait(boost::bind(increment_if_not_cancelled, &count,
        asio::arg::error));

  d.reset();
  d.run();

  // The timer should not have been cancelled, so count should have changed.
  // The total time since the timer was created should be more than 10 seconds.
  UNIT_TEST_CHECK(count == 1);
  end = asio::time::now();
  expected_end = start + 10;
  UNIT_TEST_CHECK(expected_end < end);
}

UNIT_TEST(timer_test)
