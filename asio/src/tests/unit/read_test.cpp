//
// read_test.cpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <boost/bind.hpp>
#include <boost/noncopyable.hpp>
#include <cstring>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace std; // For memcmp, memcpy and memset.

class test_stream
  : private boost::noncopyable
{
public:
  typedef asio::demuxer demuxer_type;

  test_stream(asio::demuxer& d)
    : demuxer_(d),
      length_(0),
      position_(0),
      next_read_length_(0)
  {
  }

  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  void reset(const void* data, size_t length)
  {
    UNIT_TEST_CHECK(length <= max_length);

    memcpy(data_, data, length);
    length_ = length;
    position_ = 0;
    next_read_length_ = length;
  }

  void next_read_length(size_t length)
  {
    next_read_length_ = length;
  }

  bool check(const void* data, size_t length)
  {
    if (length != position_)
      return false;

    return memcmp(data_, data, length) == 0;
  }

  size_t read(void* data, size_t length)
  {
    if (length > length_ - position_)
      length = length_ - position_;

    if (length > next_read_length_)
      length = next_read_length_;

    memcpy(data, data_ + position_, length);
    position_ += length;

    return length;
  }

  template <typename Error_Handler>
  size_t read(void* data, size_t length, Error_Handler error_handler)
  {
    if (length > length_ - position_)
      length = length_ - position_;

    if (length > next_read_length_)
      length = next_read_length_;

    memcpy(data, data_ + position_, length);
    position_ += length;

    return length;
  }

  template <typename Handler>
  void async_read(void* data, size_t length, Handler handler)
  {
    size_t bytes_transferred = read(data, length);
    asio::error error;
    demuxer_.post(
        asio::detail::bind_handler(handler, error, bytes_transferred));
  }

private:
  demuxer_type& demuxer_;
  enum { max_length = 8192 };
  char data_[max_length];
  size_t length_;
  size_t position_;
  size_t next_read_length_;
};

static const char read_data[]
  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

void test_read()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read(s, read_buf, sizeof(read_buf));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read(s, read_buf, sizeof(read_buf));
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read(s, read_buf, sizeof(read_buf));
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));
}

void test_read_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read(s, read_buf, sizeof(read_buf),
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read(s, read_buf, sizeof(read_buf),
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read(s, read_buf, sizeof(read_buf),
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));
}

void async_read_handler(const asio::error& e, size_t bytes_transferred,
    size_t expected_bytes_transferred, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(bytes_transferred == expected_bytes_transferred);
}

void test_async_read()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read(s, read_buf, sizeof(read_buf),
      boost::bind(async_read_handler, asio::placeholders::error,
        asio::placeholders::bytes_transferred, sizeof(read_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read(s, read_buf, sizeof(read_buf),
      boost::bind(async_read_handler, asio::placeholders::error,
        asio::placeholders::bytes_transferred, 1, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read(s, read_buf, sizeof(read_buf),
      boost::bind(async_read_handler, asio::placeholders::error,
        asio::placeholders::bytes_transferred, 10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, 10));
}

void test_read_n()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data),
      &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data),
      &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data),
      &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));
}

void test_read_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data),
      0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data),
      &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data), 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data),
      &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data), 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, read_buf, sizeof(read_data),
      &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));
}

void async_read_n_handler(const asio::error& e, size_t last_bytes_transferred,
    size_t total_bytes_transferred, size_t expected_last_bytes_transferred,
    size_t expected_total_bytes_transferred, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(last_bytes_transferred == expected_last_bytes_transferred);
  UNIT_TEST_CHECK(total_bytes_transferred == expected_total_bytes_transferred);
}

void test_async_read_n()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read_n(s, read_buf, sizeof(read_data),
      boost::bind(async_read_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data), sizeof(read_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_n(s, read_buf, sizeof(read_data),
      boost::bind(async_read_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 1,
        sizeof(read_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_n(s, read_buf, sizeof(read_data),
      boost::bind(async_read_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data) % 10, sizeof(read_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));
}

void test_read_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));
}

void test_read_at_least_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data), &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data), &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 1,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, 10,
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(read_data));
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));
}

void async_read_at_least_n_handler(const asio::error& e,
    size_t last_bytes_transferred, size_t total_bytes_transferred,
    size_t expected_last_bytes_transferred,
    size_t expected_total_bytes_transferred, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(last_bytes_transferred == expected_last_bytes_transferred);
  UNIT_TEST_CHECK(total_bytes_transferred == expected_total_bytes_transferred);
}

void test_async_read_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read_at_least_n(s, read_buf, 1, sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data), sizeof(read_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, read_buf, 10, sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data), sizeof(read_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data), sizeof(read_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, read_buf, 1, sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 1, 1, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, read_buf, 10, sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 1, 10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        1, sizeof(read_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, read_buf, 1, sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 10, 10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, read_buf, 10, sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 10, 10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, read_buf, sizeof(read_data),
      sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data) % 10, sizeof(read_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(read_buf, sizeof(read_data)));
}

void read_test()
{
  test_read();
  test_read_with_error_handler();
  test_async_read();

  test_read_n();
  test_read_n_with_error_handler();
  test_async_read_n();

  test_read_at_least_n();
  test_read_at_least_n_with_error_handler();
  test_async_read_at_least_n();
}

UNIT_TEST(read_test)
