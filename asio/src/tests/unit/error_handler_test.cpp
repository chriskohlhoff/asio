//
// error_handler_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/error_handler.hpp"

#include <sstream>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void error_handler_test()
{
  demuxer d;

  stream_socket s(d);
  ipv4::tcp::endpoint endpoint(321, ipv4::address::any());

  error expected_err;
  s.connect(endpoint, set_error(expected_err));
  s.close();

  std::ostringstream os;
  s.connect(endpoint, log_error(os));
  BOOST_CHECK(!os.str().empty());

  os.str("");
  s.close();
  s.connect(endpoint, the_error == expected_err || log_error(os));
  BOOST_CHECK(os.str().empty());

  os.str("");
  s.close();
  s.connect(endpoint, the_error == expected_err && log_error(os));
  BOOST_CHECK(!os.str().empty());

  os.str("");
  s.close();
  s.connect(endpoint, the_error != expected_err || log_error(os));
  BOOST_CHECK(!os.str().empty());

  os.str("");
  s.close();
  s.connect(endpoint, the_error != expected_err && log_error(os));
  BOOST_CHECK(os.str().empty());

  os.str("");
  s.close();
  s.connect(endpoint, log_error_if(os, the_error == expected_err));
  BOOST_CHECK(!os.str().empty());

  os.str("");
  s.close();
  s.connect(endpoint, log_error_if(os, the_error != expected_err));
  BOOST_CHECK(os.str().empty());

  try
  {
    s.close();
    s.connect(endpoint, throw_error());
    BOOST_CHECK(0);
  }
  catch (error&)
  {
  }

  try
  {
    s.close();
    s.connect(endpoint, the_error == expected_err || throw_error());
  }
  catch (error&)
  {
    BOOST_CHECK(0);
  }

  try
  {
    s.close();
    s.connect(endpoint, the_error == expected_err && throw_error());
    BOOST_CHECK(0);
  }
  catch (error&)
  {
  }

  try
  {
    s.close();
    s.connect(endpoint, the_error != expected_err || throw_error());
    BOOST_CHECK(0);
  }
  catch (error&)
  {
  }

  try
  {
    s.close();
    s.connect(endpoint, the_error != expected_err && throw_error());
  }
  catch (error&)
  {
    BOOST_CHECK(0);
  }

  try
  {
    s.close();
    s.connect(endpoint, throw_error_if(the_error == expected_err));
    BOOST_CHECK(0);
  }
  catch (error&)
  {
  }

  try
  {
    s.close();
    s.connect(endpoint, throw_error_if(the_error != expected_err));
  }
  catch (error&)
  {
    BOOST_CHECK(0);
  }

  s.close();
  error err;
  s.connect(endpoint, set_error(err));
  BOOST_CHECK(err == expected_err);

  s.close();
  s.connect(endpoint, the_error == expected_err || set_error(err));
  BOOST_CHECK(err != expected_err);

  s.close();
  s.connect(endpoint, the_error == expected_err && set_error(err));
  BOOST_CHECK(err == expected_err);

  s.close();
  s.connect(endpoint, the_error != expected_err || set_error(err));
  BOOST_CHECK(err == expected_err);

  s.close();
  s.connect(endpoint, the_error != expected_err && set_error(err));
  BOOST_CHECK(err != expected_err);

  s.close();
  s.connect(endpoint, set_error_if(err, the_error == expected_err));
  BOOST_CHECK(err == expected_err);

  s.close();
  s.connect(endpoint, set_error_if(err, the_error != expected_err));
  BOOST_CHECK(err != expected_err);

  try
  {
    s.close();
    s.connect(endpoint, ignore_error());
  }
  catch (error&)
  {
    BOOST_CHECK(0);
  }

  try
  {
    s.close();
    s.connect(endpoint, the_error == expected_err || ignore_error());
  }
  catch (error&)
  {
    BOOST_CHECK(0);
  }

  try
  {
    s.close();
    s.connect(endpoint, the_error == expected_err && ignore_error());
  }
  catch (error&)
  {
    BOOST_CHECK(0);
  }

  try
  {
    s.close();
    s.connect(endpoint, the_error != expected_err || ignore_error());
  }
  catch (error&)
  {
    BOOST_CHECK(0);
  }

  try
  {
    s.close();
    s.connect(endpoint, the_error != expected_err && ignore_error());
  }
  catch (error&)
  {
    BOOST_CHECK(0);
  }

  try
  {
    s.close();
    s.connect(endpoint,
        ignore_error_if(the_error == expected_err) || throw_error());
  }
  catch (error&)
  {
    BOOST_CHECK(0);
  }

  try
  {
    s.close();
    s.connect(endpoint,
        ignore_error_if(the_error != expected_err) || throw_error());
    BOOST_CHECK(0);
  }
  catch (error&)
  {
  }
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("error_handler");
  test->add(BOOST_TEST_CASE(&error_handler_test));
  return test;
}
