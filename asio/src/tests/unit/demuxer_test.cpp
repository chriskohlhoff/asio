//
// demuxer_test.hpp
// ~~~~~~~~~~~~~~~~
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

#include <sstream>
#include <boost/bind.hpp>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void increment(int* count)
{
  ++(*count);
}

void decrement_to_zero(demuxer* d, int* count)
{
  if (*count > 0)
  {
    --(*count);

    int before_value = *count;
    d->operation_immediate(boost::bind(decrement_to_zero, d, count));

    // Completion cannot nest, so count value should remain unchanged.
    UNIT_TEST_CHECK(*count == before_value);
  }
}

void nested_decrement_to_zero(demuxer* d, int* count)
{
  if (*count > 0)
  {
    --(*count);

    d->operation_immediate(boost::bind(nested_decrement_to_zero, d, count),
        null_completion_context(), true);

    // Completion is nested, so count value should now be zero.
    UNIT_TEST_CHECK(*count == 0);
  }
}

void sleep_increment(demuxer* d, int* count)
{
  timer t(*d, timer::from_now, 2);
  t.wait();

  ++(*count);
}

void start_sleep_increments(demuxer* d, int* count)
{
  // Give all threads a chance to start.
  timer t(*d, timer::from_now, 2);
  t.wait();

  // Start three increments which cannot run in parallel.
  counting_completion_context ctx1(1);
  d->operation_immediate(boost::bind(sleep_increment, d, count), ctx1);
  d->operation_immediate(boost::bind(sleep_increment, d, count), ctx1);
  d->operation_immediate(boost::bind(sleep_increment, d, count), ctx1);
}

void demuxer_test()
{
  demuxer d;
  int count = 0;

  d.operation_immediate(boost::bind(increment, &count));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all operations have finished.
  UNIT_TEST_CHECK(count == 1);

  count = 0;
  d.reset();
  d.operation_immediate(boost::bind(increment, &count));
  d.operation_immediate(boost::bind(increment, &count));
  d.operation_immediate(boost::bind(increment, &count));
  d.operation_immediate(boost::bind(increment, &count));
  d.operation_immediate(boost::bind(increment, &count));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all operations have finished.
  UNIT_TEST_CHECK(count == 5);

  count = 0;
  d.reset();
  d.operation_started();
  d.operation_immediate(boost::bind(&demuxer::interrupt, &d));
  d.run();

  // The only operation executed should have been to interrupt run().
  UNIT_TEST_CHECK(count == 0);

  d.reset();
  d.operation_completed(boost::bind(increment, &count));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all operations have finished.
  UNIT_TEST_CHECK(count == 1);

  count = 10;
  d.reset();
  d.operation_immediate(boost::bind(decrement_to_zero, &d, &count));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 10);

  d.run();

  // The run() call will not return until all operations have finished.
  UNIT_TEST_CHECK(count == 0);

  count = 10;
  d.reset();
  d.operation_immediate(boost::bind(nested_decrement_to_zero, &d, &count));

  // No completions can be delivered until run() is called.
  UNIT_TEST_CHECK(count == 10);

  d.run();

  // The run() call will not return until all operations have finished.
  UNIT_TEST_CHECK(count == 0);

  count = 10;
  d.reset();
  d.operation_immediate(boost::bind(nested_decrement_to_zero, &d, &count),
      null_completion_context(), true);

  // No completions can be delivered until run() is called, even though nested
  // delivery was specifically allowed in the previous call.
  UNIT_TEST_CHECK(count == 10);

  d.run();

  // The run() call will not return until all operations have finished.
  UNIT_TEST_CHECK(count == 0);

  count = 0;
  d.reset();
  d.operation_immediate(boost::bind(start_sleep_increments, &d, &count));
  detail::thread thread1(boost::bind(&demuxer::run, &d));
  detail::thread thread2(boost::bind(&demuxer::run, &d));

  // Check all events run one after another even though there are two threads.
  timer timer1(d, timer::from_now, 3);
  timer1.wait();
  UNIT_TEST_CHECK(count == 0);
  timer1.set(timer::from_existing, 2);
  timer1.wait();
  UNIT_TEST_CHECK(count == 1);
  timer1.set(timer::from_existing, 2);
  timer1.wait();
  UNIT_TEST_CHECK(count == 2);

  thread1.join();
  thread2.join();

  // The run() calls will not return until all operations have finished.
  UNIT_TEST_CHECK(count == 3);
}

UNIT_TEST(demuxer_test)
