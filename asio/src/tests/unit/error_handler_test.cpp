#include <sstream>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace asio;

void error_handler_test()
{
  demuxer d;

  stream_socket s(d);
  socket_connector c(d);

  socket_error expected_err;
  c.connect(s, inet_address_v4(321, "0.0.0.0"), set_error(expected_err));

  std::ostringstream os;
  c.connect(s, inet_address_v4(321, "0.0.0.0"), log_error(os));
  UNIT_TEST_CHECK(!os.str().empty());

  os.str("");
  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      error == expected_err || log_error(os));
  UNIT_TEST_CHECK(os.str().empty());

  os.str("");
  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      error == expected_err && log_error(os));
  UNIT_TEST_CHECK(!os.str().empty());

  os.str("");
  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      error != expected_err || log_error(os));
  UNIT_TEST_CHECK(!os.str().empty());

  os.str("");
  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      error != expected_err && log_error(os));
  UNIT_TEST_CHECK(os.str().empty());

  os.str("");
  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      log_error_if(os, error == expected_err));
  UNIT_TEST_CHECK(!os.str().empty());

  os.str("");
  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      log_error_if(os, error != expected_err));
  UNIT_TEST_CHECK(os.str().empty());

  try
  {
    c.connect(s, inet_address_v4(321, "0.0.0.0"), throw_error());
    UNIT_TEST_CHECK(0);
  }
  catch (socket_error&)
  {
  }

  try
  {
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error == expected_err || throw_error());
  }
  catch (socket_error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error == expected_err && throw_error());
    UNIT_TEST_CHECK(0);
  }
  catch (socket_error&)
  {
  }

  try
  {
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error != expected_err || throw_error());
    UNIT_TEST_CHECK(0);
  }
  catch (socket_error&)
  {
  }

  try
  {
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        error != expected_err && throw_error());
  }
  catch (socket_error&)
  {
    UNIT_TEST_CHECK(0);
  }

  try
  {
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        throw_error_if(error == expected_err));
    UNIT_TEST_CHECK(0);
  }
  catch (socket_error&)
  {
  }

  try
  {
    c.connect(s, inet_address_v4(321, "0.0.0.0"),
        throw_error_if(error != expected_err));
  }
  catch (socket_error&)
  {
    UNIT_TEST_CHECK(0);
  }

  socket_error err;
  c.connect(s, inet_address_v4(321, "0.0.0.0"), set_error(err));
  UNIT_TEST_CHECK(err == expected_err);

  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      error == expected_err || set_error(err));
  UNIT_TEST_CHECK(err != expected_err);

  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      error == expected_err && set_error(err));
  UNIT_TEST_CHECK(err == expected_err);

  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      error != expected_err || set_error(err));
  UNIT_TEST_CHECK(err == expected_err);

  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      error != expected_err && set_error(err));
  UNIT_TEST_CHECK(err != expected_err);

  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      set_error_if(err, error == expected_err));
  UNIT_TEST_CHECK(err == expected_err);

  c.connect(s, inet_address_v4(321, "0.0.0.0"),
      set_error_if(err, error != expected_err));
  UNIT_TEST_CHECK(err != expected_err);
}

UNIT_TEST(error_handler_test)
