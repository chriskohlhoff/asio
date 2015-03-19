//
// io_service.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/io_service.hpp"

#include <sstream>
#include "asio/thread.hpp"
#include "unit_test.hpp"

#if defined(ASIO_HAS_BOOST_DATE_TIME)
# include "asio/deadline_timer.hpp"
#else // defined(ASIO_HAS_BOOST_DATE_TIME)
# include "asio/steady_timer.hpp"
#endif // defined(ASIO_HAS_BOOST_DATE_TIME)

#if defined(ASIO_HAS_BOOST_BIND)
# include <boost/bind.hpp>
#else // defined(ASIO_HAS_BOOST_BIND)
# include <functional>
#endif // defined(ASIO_HAS_BOOST_BIND)

using namespace asio;

#if defined(ASIO_HAS_BOOST_BIND)
namespace bindns = boost;
#else // defined(ASIO_HAS_BOOST_BIND)
namespace bindns = std;
#endif

#if defined(ASIO_HAS_BOOST_DATE_TIME)
typedef deadline_timer timer;
namespace chronons = boost::posix_time;
#elif defined(ASIO_HAS_STD_CHRONO)
typedef steady_timer timer;
namespace chronons = std::chrono;
#elif defined(ASIO_HAS_BOOST_CHRONO)
typedef steady_timer timer;
namespace chronons = boost::chrono;
#endif // defined(ASIO_HAS_BOOST_DATE_TIME)

void increment(int* count)
{
  ++(*count);
}

void decrement_to_zero(io_service* ios, int* count)
{
  if (*count > 0)
  {
    --(*count);

    int before_value = *count;
    ios->post(bindns::bind(decrement_to_zero, ios, count));

    // Handler execution cannot nest, so count value should remain unchanged.
    ASIO_CHECK(*count == before_value);
  }
}

void nested_decrement_to_zero(io_service* ios, int* count)
{
  if (*count > 0)
  {
    --(*count);

    ios->dispatch(bindns::bind(nested_decrement_to_zero, ios, count));

    // Handler execution is nested, so count value should now be zero.
    ASIO_CHECK(*count == 0);
  }
}

void sleep_increment(io_service* ios, int* count)
{
  timer t(*ios, chronons::seconds(2));
  t.wait();

  if (++(*count) < 3)
    ios->post(bindns::bind(sleep_increment, ios, count));
}

void start_sleep_increments(io_service* ios, int* count)
{
  // Give all threads a chance to start.
  timer t(*ios, chronons::seconds(2));
  t.wait();

  // Start the first of three increments.
  ios->post(bindns::bind(sleep_increment, ios, count));
}

void throw_exception()
{
  throw 1;
}

void io_service_run(io_service* ios)
{
  ios->run();
}

void io_service_test()
{
  io_service ios;
  int count = 0;

  ios.post(bindns::bind(increment, &count));

  // No handlers can be called until run() is called.
  ASIO_CHECK(!ios.stopped());
  ASIO_CHECK(count == 0);

  ios.run();

  // The run() call will not return until all work has finished.
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 1);

  count = 0;
  ios.reset();
  ios.post(bindns::bind(increment, &count));
  ios.post(bindns::bind(increment, &count));
  ios.post(bindns::bind(increment, &count));
  ios.post(bindns::bind(increment, &count));
  ios.post(bindns::bind(increment, &count));

  // No handlers can be called until run() is called.
  ASIO_CHECK(!ios.stopped());
  ASIO_CHECK(count == 0);

  ios.run();

  // The run() call will not return until all work has finished.
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 5);

  count = 0;
  ios.reset();
  io_service::work* w = new io_service::work(ios);
  ios.post(bindns::bind(&io_service::stop, &ios));
  ASIO_CHECK(!ios.stopped());
  ios.run();

  // The only operation executed should have been to stop run().
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 0);

  ios.reset();
  ios.post(bindns::bind(increment, &count));
  delete w;

  // No handlers can be called until run() is called.
  ASIO_CHECK(!ios.stopped());
  ASIO_CHECK(count == 0);

  ios.run();

  // The run() call will not return until all work has finished.
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 1);

  count = 10;
  ios.reset();
  ios.post(bindns::bind(decrement_to_zero, &ios, &count));

  // No handlers can be called until run() is called.
  ASIO_CHECK(!ios.stopped());
  ASIO_CHECK(count == 10);

  ios.run();

  // The run() call will not return until all work has finished.
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 0);

  count = 10;
  ios.reset();
  ios.post(bindns::bind(nested_decrement_to_zero, &ios, &count));

  // No handlers can be called until run() is called.
  ASIO_CHECK(!ios.stopped());
  ASIO_CHECK(count == 10);

  ios.run();

  // The run() call will not return until all work has finished.
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 0);

  count = 10;
  ios.reset();
  ios.dispatch(bindns::bind(nested_decrement_to_zero, &ios, &count));

  // No handlers can be called until run() is called, even though nested
  // delivery was specifically allowed in the previous call.
  ASIO_CHECK(!ios.stopped());
  ASIO_CHECK(count == 10);

  ios.run();

  // The run() call will not return until all work has finished.
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 0);

  count = 0;
  int count2 = 0;
  ios.reset();
  ASIO_CHECK(!ios.stopped());
  ios.post(bindns::bind(start_sleep_increments, &ios, &count));
  ios.post(bindns::bind(start_sleep_increments, &ios, &count2));
  thread thread1(bindns::bind(io_service_run, &ios));
  thread thread2(bindns::bind(io_service_run, &ios));
  thread1.join();
  thread2.join();

  // The run() calls will not return until all work has finished.
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 3);
  ASIO_CHECK(count2 == 3);

  count = 10;
  io_service ios2;
  ios.dispatch(ios2.wrap(bindns::bind(decrement_to_zero, &ios2, &count)));
  ios.reset();
  ASIO_CHECK(!ios.stopped());
  ios.run();

  // No decrement_to_zero handlers can be called until run() is called on the
  // second io_service object.
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 10);

  ios2.run();

  // The run() call will not return until all work has finished.
  ASIO_CHECK(count == 0);

  count = 0;
  int exception_count = 0;
  ios.reset();
  ios.post(&throw_exception);
  ios.post(bindns::bind(increment, &count));
  ios.post(bindns::bind(increment, &count));
  ios.post(&throw_exception);
  ios.post(bindns::bind(increment, &count));

  // No handlers can be called until run() is called.
  ASIO_CHECK(!ios.stopped());
  ASIO_CHECK(count == 0);
  ASIO_CHECK(exception_count == 0);

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
  ASIO_CHECK(ios.stopped());
  ASIO_CHECK(count == 3);
  ASIO_CHECK(exception_count == 2);
}

class test_service : public asio::io_service::service
{
public:
  static asio::io_service::id id;
  test_service(asio::io_service& s)
    : asio::io_service::service(s) {}
private:
  virtual void shutdown_service() {}
};

asio::io_service::id test_service::id;

void io_service_service_test()
{
  asio::io_service ios1;
  asio::io_service ios2;
  asio::io_service ios3;

  // Implicit service registration.

  asio::use_service<test_service>(ios1);

  ASIO_CHECK(asio::has_service<test_service>(ios1));

  test_service* svc1 = new test_service(ios1);
  try
  {
    asio::add_service(ios1, svc1);
    ASIO_ERROR("add_service did not throw");
  }
  catch (asio::service_already_exists&)
  {
  }
  delete svc1;

  // Explicit service registration.

  test_service* svc2 = new test_service(ios2);
  asio::add_service(ios2, svc2);

  ASIO_CHECK(asio::has_service<test_service>(ios2));
  ASIO_CHECK(&asio::use_service<test_service>(ios2) == svc2);

  test_service* svc3 = new test_service(ios2);
  try
  {
    asio::add_service(ios2, svc3);
    ASIO_ERROR("add_service did not throw");
  }
  catch (asio::service_already_exists&)
  {
  }
  delete svc3;

  // Explicit registration with invalid owner.

  test_service* svc4 = new test_service(ios2);
  try
  {
    asio::add_service(ios3, svc4);
    ASIO_ERROR("add_service did not throw");
  }
  catch (asio::invalid_service_owner&)
  {
  }
  delete svc4;

  ASIO_CHECK(!asio::has_service<test_service>(ios3));
}

ASIO_TEST_SUITE
(
  "io_service",
  ASIO_TEST_CASE(io_service_test)
  ASIO_TEST_CASE(io_service_service_test)
)
