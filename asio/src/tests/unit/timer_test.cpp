//
// timer_test.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#include <boost/bind.hpp>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void increment(int* count)
{
  ++(*count);
}

void decrement_to_zero(timer* t, int* count)
{
  if (*count > 0)
  {
    --(*count);

    int before_value = *count;

    t->set(timer::from_existing, 1);
    t->async_wait(boost::bind(decrement_to_zero, t, count));

    // Completion cannot nest, so count value should remain unchanged.
    UNIT_TEST_CHECK(*count == before_value);
  }
}

void increment_if_not_cancelled(int* count, const error& e)
{
  if (!e)
    ++(*count);
}

void cancel_timer(timer* t)
{
  int num_cancelled = t->cancel();
  UNIT_TEST_CHECK(num_cancelled == 1);
}

void timer_test()
{
  demuxer d;
  int count = 0;

  detail::time start = detail::time::now();

  timer t1(d, timer::from_now, 1);
  t1.wait();

  // The timer must block until after its expiry time.
  detail::time end = detail::time::now();
  detail::time expected_end = start;
  expected_end += detail::time(1);
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  start = detail::time::now();

  timer t2(d, timer::from_now, 1, 500000);
  t2.wait();

  // The timer must block until after its expiry time.
  end = detail::time::now();
  expected_end = start;
  expected_end += detail::time(1, 500000);
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  t2.set(timer::from_existing, 1);
  t2.wait();

  // The timer must block until after its expiry time.
  end = detail::time::now();
  expected_end += detail::time(1);
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  t2.set(timer::from_existing, 1, 200000);
  t2.wait();

  // The timer must block until after its expiry time.
  end = detail::time::now();
  expected_end += detail::time(1, 200000);
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  start = detail::time::now();

  timer t3(d, timer::from_now, 1);
  t3.async_wait(boost::bind(increment, &count));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all operations have finished, and
  // this should not be until after the timer's expiry time.
  UNIT_TEST_CHECK(count == 1);
  end = detail::time::now();
  expected_end = start;
  expected_end += detail::time(1);
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  count = 3;
  start = detail::time::now();

  timer t4(d, timer::from_now, 1);
  t4.async_wait(boost::bind(decrement_to_zero, &t4, &count));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 3);

  d.reset();
  d.run();

  // The run() call will not return until all operations have finished, and
  // this should not be until after the timer's final expiry time.
  UNIT_TEST_CHECK(count == 0);
  end = detail::time::now();
  expected_end = start;
  expected_end += detail::time(3);
  UNIT_TEST_CHECK(expected_end < end || expected_end == end);

  count = 0;
  start = detail::time::now();

  timer t5(d, timer::from_now, 10);
  t5.async_wait(boost::bind(increment_if_not_cancelled, &count, arg::error));
  timer t6(d, timer::from_now, 1);
  t6.async_wait(boost::bind(cancel_timer, &t5));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.reset();
  d.run();

  // The timer should have been cancelled, so count should not have changed.
  // The total run time should not have been much more than 1 second (and
  // certainly far less than 10 seconds).
  UNIT_TEST_CHECK(count == 0);
  end = detail::time::now();
  expected_end = start;
  expected_end += detail::time(2);
  UNIT_TEST_CHECK(end < expected_end);

  // Wait on the timer again without cancelling it. This time the asynchronous
  // wait should run to completion and increment the counter.
  t5.async_wait(boost::bind(increment_if_not_cancelled, &count, arg::error));

  d.reset();
  d.run();

  // The timer should not have been cancelled, so count should have changed.
  // The total time since the timer was created should be more than 10 seconds.
  UNIT_TEST_CHECK(count == 1);
  end = detail::time::now();
  expected_end = start;
  expected_end += detail::time(10);
  UNIT_TEST_CHECK(expected_end < end);
}

UNIT_TEST(timer_test)
