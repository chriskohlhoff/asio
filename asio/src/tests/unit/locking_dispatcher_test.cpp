//
// locking_dispatcher_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

void increment_without_lock(locking_dispatcher* l, int* count)
{
  int original_count = *count;

  l->dispatch(boost::bind(increment, count));

  // No other functions are currently executing through the locking dispatcher,
  // so the previous call to dispatch should have successfully nested.
  UNIT_TEST_CHECK(*count == original_count + 1);
}

void increment_with_lock(locking_dispatcher* l, int* count)
{
  int original_count = *count;

  l->dispatch(boost::bind(increment, count));

  // The current function already holds the locking_dispatcher's lock, so the
  // previous call to dispatch should not have nested.
  UNIT_TEST_CHECK(*count == original_count);
}

void sleep_increment(demuxer* d, int* count)
{
  timer t(*d, asio::time::now() + 2);
  t.wait();

  ++(*count);
}

void start_sleep_increments(demuxer* d, locking_dispatcher* l, int* count)
{
  // Give all threads a chance to start.
  timer t(*d, asio::time::now() + 2);
  t.wait();

  // Start three increments.
  l->post(boost::bind(sleep_increment, d, count));
  l->post(boost::bind(sleep_increment, d, count));
  l->post(boost::bind(sleep_increment, d, count));
}

void locking_dispatcher_test()
{
  demuxer d;
  locking_dispatcher l(d);
  int count = 0;

  d.post(boost::bind(increment_without_lock, &l, &count));

  // No handlers can be called until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all work has finished.
  UNIT_TEST_CHECK(count == 1);

  count = 0;
  d.reset();
  l.post(boost::bind(increment_with_lock, &l, &count));

  // No handlers can be called until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all work has finished.
  UNIT_TEST_CHECK(count == 1);

  count = 0;
  d.reset();
  d.post(boost::bind(start_sleep_increments, &d, &l, &count));
  thread thread1(boost::bind(&demuxer::run, &d));
  thread thread2(boost::bind(&demuxer::run, &d));

  // Check all events run one after another even though there are two threads.
  timer timer1(d, asio::time::now() + 3);
  timer1.wait();
  UNIT_TEST_CHECK(count == 0);
  timer1.expiry(timer1.expiry() + 2);
  timer1.wait();
  UNIT_TEST_CHECK(count == 1);
  timer1.expiry(timer1.expiry() + 2);
  timer1.wait();
  UNIT_TEST_CHECK(count == 2);

  thread1.join();
  thread2.join();

  // The run() calls will not return until all work has finished.
  UNIT_TEST_CHECK(count == 3);
}

UNIT_TEST(locking_dispatcher_test)
