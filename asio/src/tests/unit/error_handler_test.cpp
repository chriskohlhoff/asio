//
// error_handler_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~
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
#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void error_handler_test()
{
  demuxer d;

  stream_socket s(d);
  socket_connector c(d);
  ipv4::tcp::endpoint endpoint(321, ipv4::address::any());

  error expected_err;
  c.connect(s, endpoint, set_error(expected_err));

  std::ostringstream os;
  c.connect(s, endpoint, log_error(os));
  UNIT_TEST_CHECK(!os.str().empty());

  os.str("");
  c.connect(s, endpoint, the_error == expected_err || log_error(os));
  UNIT_TEST_CHECK(os.str().empty());

  os.str("");
  c.connect(s, endpoint, the_error == expected_err && log_error(os));
  UNIT_TEST_CHECK(!os.str().empty());

  os.str("");
  c.connect(s, endpoint, the_error != expected_err || log_error(os));
  UNIT_TEST_CHECK(!os.str().empty());

  os.str("");
  c.connect(s, endpoint, the_error != expected_err && log_error(os));
  UNIT_TEST_CHECK(os.str().empty());

  os.str("");
  c.connect(s, endpoint, log_error_if(os, the_error == expected_err));
  UNIT_TEST_CHECK(!os.str().empty());

  os.str("");
  c.connect(s, endpoint, log_error_if(os, the_error != expected_err));
  UNIT_TEST_CHECK(os.str().empty());

  try
  {
    c.connect(s, endpoint, throw_error());
    UNIT_TEST_CHECK(0);
  }
  catch (error&)
  {
  }

  try
  {
    c.connect(s, endpoint, the_error == expected_err || throw_error());
  }
  catch (error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, endpoint, the_error == expected_err && throw_error());
    UNIT_TEST_CHECK(0);
  }
  catch (error&)
  {
  }

  try
  {
    c.connect(s, endpoint, the_error != expected_err || throw_error());
    UNIT_TEST_CHECK(0);
  }
  catch (error&)
  {
  }

  try
  {
    c.connect(s, endpoint, the_error != expected_err && throw_error());
  }
  catch (error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, endpoint, throw_error_if(the_error == expected_err));
    UNIT_TEST_CHECK(0);
  }
  catch (error&)
  {
  }

  try
  {
    c.connect(s, endpoint, throw_error_if(the_error != expected_err));
  }
  catch (error&)
  {
    UNIT_TEST_CHECK(0);
  }

  error err;
  c.connect(s, endpoint, set_error(err));
  UNIT_TEST_CHECK(err == expected_err);

  c.connect(s, endpoint, the_error == expected_err || set_error(err));
  UNIT_TEST_CHECK(err != expected_err);

  c.connect(s, endpoint, the_error == expected_err && set_error(err));
  UNIT_TEST_CHECK(err == expected_err);

  c.connect(s, endpoint, the_error != expected_err || set_error(err));
  UNIT_TEST_CHECK(err == expected_err);

  c.connect(s, endpoint, the_error != expected_err && set_error(err));
  UNIT_TEST_CHECK(err != expected_err);

  c.connect(s, endpoint, set_error_if(err, the_error == expected_err));
  UNIT_TEST_CHECK(err == expected_err);

  c.connect(s, endpoint, set_error_if(err, the_error != expected_err));
  UNIT_TEST_CHECK(err != expected_err);

  try
  {
    c.connect(s, endpoint, ignore_error());
  }
  catch (error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, endpoint, the_error == expected_err || ignore_error());
  }
  catch (error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, endpoint, the_error == expected_err && ignore_error());
  }
  catch (error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, endpoint, the_error != expected_err || ignore_error());
  }
  catch (error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, endpoint, the_error != expected_err && ignore_error());
  }
  catch (error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, endpoint,
        ignore_error_if(the_error == expected_err) || throw_error());
  }
  catch (error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, endpoint,
        ignore_error_if(the_error != expected_err) || throw_error());
    UNIT_TEST_CHECK(0);
  }
  catch (error&)
  {
  }
}

UNIT_TEST(error_handler_test)
