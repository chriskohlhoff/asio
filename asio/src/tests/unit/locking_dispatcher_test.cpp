//
// locking_dispatcher_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/locking_dispatcher.hpp"

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
  BOOST_CHECK(*count == original_count + 1);
}

void increment_with_lock(locking_dispatcher* l, int* count)
{
  int original_count = *count;

  l->dispatch(boost::bind(increment, count));

  // The current function already holds the locking_dispatcher's lock, so the
  // previous call to dispatch should not have nested.
  BOOST_CHECK(*count == original_count);
}

void sleep_increment(io_service* ios, int* count)
{
  deadline_timer t(*ios, boost::posix_time::seconds(2));
  t.wait();

  ++(*count);
}

void start_sleep_increments(io_service* ios, locking_dispatcher* l, int* count)
{
  // Give all threads a chance to start.
  deadline_timer t(*ios, boost::posix_time::seconds(2));
  t.wait();

  // Start three increments.
  l->post(boost::bind(sleep_increment, ios, count));
  l->post(boost::bind(sleep_increment, ios, count));
  l->post(boost::bind(sleep_increment, ios, count));
}

void throw_exception()
{
  throw 1;
}

void locking_dispatcher_test()
{
  io_service ios;
  locking_dispatcher l(ios);
  int count = 0;

  ios.post(boost::bind(increment_without_lock, &l, &count));

  // No handlers can be called until run() is called.
  BOOST_CHECK(count == 0);

  ios.run();

  // The run() call will not return until all work has finished.
  BOOST_CHECK(count == 1);

  count = 0;
  ios.reset();
  l.post(boost::bind(increment_with_lock, &l, &count));

  // No handlers can be called until run() is called.
  BOOST_CHECK(count == 0);

  ios.run();

  // The run() call will not return until all work has finished.
  BOOST_CHECK(count == 1);

  count = 0;
  ios.reset();
  ios.post(boost::bind(start_sleep_increments, &ios, &l, &count));
  thread thread1(boost::bind(&io_service::run, &ios));
  thread thread2(boost::bind(&io_service::run, &ios));

  // Check all events run one after another even though there are two threads.
  deadline_timer timer1(ios, boost::posix_time::seconds(3));
  timer1.wait();
  BOOST_CHECK(count == 0);
  timer1.expires_at(timer1.expires_at() + boost::posix_time::seconds(2));
  timer1.wait();
  BOOST_CHECK(count == 1);
  timer1.expires_at(timer1.expires_at() + boost::posix_time::seconds(2));
  timer1.wait();
  BOOST_CHECK(count == 2);

  thread1.join();
  thread2.join();

  // The run() calls will not return until all work has finished.
  BOOST_CHECK(count == 3);

  count = 0;
  int exception_count = 0;
  ios.reset();
  l.post(throw_exception);
  l.post(boost::bind(increment, &count));
  l.post(boost::bind(increment, &count));
  l.post(throw_exception);
  l.post(boost::bind(increment, &count));

  // No handlers can be called until run() is called.
  BOOST_CHECK(count == 0);
  BOOST_CHECK(exception_count == 0);

  for (;;)
  {
    try
    {
      ios.run();
      break;
    }
    catch (int)
    {
      ++exception_count;
    }
  }

  // The run() calls will not return until all work has finished.
  BOOST_CHECK(count == 3);
  BOOST_CHECK(exception_count == 2);
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("locking_dispatcher");
  test->add(BOOST_TEST_CASE(&locking_dispatcher_test));
  return test;
}
