//
// demuxer_test.cpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
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
    d->post(boost::bind(decrement_to_zero, d, count));

    // Handler execution cannot nest, so count value should remain unchanged.
    UNIT_TEST_CHECK(*count == before_value);
  }
}

void nested_decrement_to_zero(demuxer* d, int* count)
{
  if (*count > 0)
  {
    --(*count);

    d->dispatch(boost::bind(nested_decrement_to_zero, d, count));

    // Handler execution is nested, so count value should now be zero.
    UNIT_TEST_CHECK(*count == 0);
  }
}

void sleep_increment(demuxer* d, int* count)
{
  deadline_timer t(*d, boost::posix_time::seconds(2));
  t.wait();

  if (++(*count) < 3)
    d->post(boost::bind(sleep_increment, d, count));
}

void start_sleep_increments(demuxer* d, int* count)
{
  // Give all threads a chance to start.
  deadline_timer t(*d, boost::posix_time::seconds(2));
  t.wait();

  // Start the first of three increments.
  d->post(boost::bind(sleep_increment, d, count));
}

void demuxer_test()
{
  demuxer d;
  int count = 0;

  d.post(boost::bind(increment, &count));

  // No handlers can be called until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all work has finished.
  UNIT_TEST_CHECK(count == 1);

  count = 0;
  d.reset();
  d.post(boost::bind(increment, &count));
  d.post(boost::bind(increment, &count));
  d.post(boost::bind(increment, &count));
  d.post(boost::bind(increment, &count));
  d.post(boost::bind(increment, &count));

  // No handlers can be called until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all work has finished.
  UNIT_TEST_CHECK(count == 5);

  count = 0;
  d.reset();
  d.work_started();
  d.post(boost::bind(&demuxer::interrupt, &d));
  d.run();

  // The only operation executed should have been to interrupt run().
  UNIT_TEST_CHECK(count == 0);

  d.reset();
  d.post(boost::bind(increment, &count));
  d.work_finished();

  // No handlers can be called until run() is called.
  UNIT_TEST_CHECK(count == 0);

  d.run();

  // The run() call will not return until all work has finished.
  UNIT_TEST_CHECK(count == 1);

  count = 10;
  d.reset();
  d.post(boost::bind(decrement_to_zero, &d, &count));

  // No handlers can be called until run() is called.
  UNIT_TEST_CHECK(count == 10);

  d.run();

  // The run() call will not return until all work has finished.
  UNIT_TEST_CHECK(count == 0);

  count = 10;
  d.reset();
  d.post(boost::bind(nested_decrement_to_zero, &d, &count));

  // No handlers can be called until run() is called.
  UNIT_TEST_CHECK(count == 10);

  d.run();

  // The run() call will not return until all work has finished.
  UNIT_TEST_CHECK(count == 0);

  count = 10;
  d.reset();
  d.dispatch(boost::bind(nested_decrement_to_zero, &d, &count));

  // No handlers can be called until run() is called, even though nested
  // delivery was specifically allowed in the previous call.
  UNIT_TEST_CHECK(count == 10);

  d.run();

  // The run() call will not return until all work has finished.
  UNIT_TEST_CHECK(count == 0);

  count = 0;
  int count2 = 0;
  d.reset();
  d.post(boost::bind(start_sleep_increments, &d, &count));
  d.post(boost::bind(start_sleep_increments, &d, &count2));
  thread thread1(boost::bind(&demuxer::run, &d));
  thread thread2(boost::bind(&demuxer::run, &d));
  thread1.join();
  thread2.join();

  // The run() calls will not return until all work has finished.
  UNIT_TEST_CHECK(count == 3);
  UNIT_TEST_CHECK(count2 == 3);

  count = 10;
  demuxer d2;
  d.dispatch(d2.wrap(boost::bind(decrement_to_zero, &d2, &count)));
  d.reset();
  d.run();

  // No decrement_to_zero handlers can be called until run() is called on the
  // second demuxer object.
  UNIT_TEST_CHECK(count == 10);

  d2.run();

  // The run() call will not return until all work has finished.
  UNIT_TEST_CHECK(count == 0);

#if !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
  // Use a non-default allocator type.
  typedef std::allocator<int> allocator_type;
  typedef demuxer_service<allocator_type> demuxer_service_type;
  typedef basic_demuxer<demuxer_service_type> demuxer_type;
  allocator_type allocator;
  service_factory<demuxer_service_type> factory(allocator);
  demuxer_type d3(factory);
  d3.run();
#endif // !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
}

UNIT_TEST(demuxer_test)
