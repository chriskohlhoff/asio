//
// recv_test.cpp
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
      next_recv_length_(0)
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
    next_recv_length_ = length;
  }

  void next_recv_length(size_t length)
  {
    next_recv_length_ = length;
  }

  bool check(const void* data, size_t length)
  {
    if (length != position_)
      return false;

    return memcmp(data_, data, length) == 0;
  }

  size_t recv(void* data, size_t length)
  {
    if (length > length_ - position_)
      length = length_ - position_;

    if (length > next_recv_length_)
      length = next_recv_length_;

    memcpy(data, data_ + position_, length);
    position_ += length;

    return length;
  }

  template <typename Error_Handler>
  size_t recv(void* data, size_t length, Error_Handler error_handler)
  {
    if (length > length_ - position_)
      length = length_ - position_;

    if (length > next_recv_length_)
      length = next_recv_length_;

    memcpy(data, data_ + position_, length);
    position_ += length;

    return length;
  }

  template <typename Handler>
  void async_recv(void* data, size_t length, Handler handler)
  {
    size_t bytes_recvd = recv(data, length);
    asio::error error;
    demuxer_.post(asio::detail::bind_handler(handler, error, bytes_recvd));
  }

private:
  demuxer_type& demuxer_;
  enum { max_length = 8192 };
  char data_[max_length];
  size_t length_;
  size_t position_;
  size_t next_recv_length_;
};

static const char recv_data[]
  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

void test_recv()
{
  asio::demuxer d;
  test_stream s(d);
  char recv_buf[1024];

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t last_bytes_recvd = asio::recv(s, recv_buf, sizeof(recv_buf));
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv(s, recv_buf, sizeof(recv_buf));
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, 1));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv(s, recv_buf, sizeof(recv_buf));
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));
}

void test_recv_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  char recv_buf[1024];

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t last_bytes_recvd = asio::recv(s, recv_buf, sizeof(recv_buf),
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv(s, recv_buf, sizeof(recv_buf),
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, 1));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv(s, recv_buf, sizeof(recv_buf),
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));
}

void async_recv_handler(const asio::error& e, size_t bytes_recvd,
    size_t expected_bytes_recvd, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(bytes_recvd == expected_bytes_recvd);
}

void test_async_recv()
{
  asio::demuxer d;
  test_stream s(d);
  char recv_buf[1024];

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  bool called = false;
  asio::async_recv(s, recv_buf, sizeof(recv_buf),
      boost::bind(async_recv_handler, asio::arg::error, asio::arg::bytes_recvd,
        sizeof(recv_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv(s, recv_buf, sizeof(recv_buf),
      boost::bind(async_recv_handler, asio::arg::error, asio::arg::bytes_recvd,
        1, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, 1));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv(s, recv_buf, sizeof(recv_buf),
      boost::bind(async_recv_handler, asio::arg::error, asio::arg::bytes_recvd,
        10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));
}

void test_recv_n()
{
  asio::demuxer d;
  test_stream s(d);
  char recv_buf[1024];

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data) % 10);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data) % 10);
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));
}

void test_recv_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  char recv_buf[1024];

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data), 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data),
      &total_bytes_recvd, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data), 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data),
      &total_bytes_recvd, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data), 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data) % 10);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_n(s, recv_buf, sizeof(recv_data),
      &total_bytes_recvd, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data) % 10);
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));
}

void async_recv_n_handler(const asio::error& e, size_t last_bytes_recvd,
    size_t total_bytes_recvd, size_t expected_last_bytes_recvd,
    size_t expected_total_bytes_recvd, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(last_bytes_recvd == expected_last_bytes_recvd);
  UNIT_TEST_CHECK(total_bytes_recvd == expected_total_bytes_recvd);
}

void test_async_recv_n()
{
  asio::demuxer d;
  test_stream s(d);
  char recv_buf[1024];

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  bool called = false;
  asio::async_recv_n(s, recv_buf, sizeof(recv_data),
      boost::bind(async_recv_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd,
        sizeof(recv_data), sizeof(recv_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_n(s, recv_buf, sizeof(recv_data),
      boost::bind(async_recv_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd, 1,
        sizeof(recv_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_n(s, recv_buf, sizeof(recv_data),
      boost::bind(async_recv_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd,
        sizeof(recv_data) % 10, sizeof(recv_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));
}

void test_recv_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);
  char recv_buf[1024];

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1,
      sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data), &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, 1));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(total_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, 1));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(total_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data), &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(total_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(total_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data) % 10);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data), &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data) % 10);
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));
}

void test_recv_at_least_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  char recv_buf[1024];

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1,
      sizeof(recv_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  size_t total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data),
      &total_bytes_recvd, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      &total_bytes_recvd, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data), &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, 1));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(total_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, 1));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(total_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data), &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 1);
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 1, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(total_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == 10);
  UNIT_TEST_CHECK(total_bytes_recvd == 10);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data));
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data) % 10);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  total_bytes_recvd = 0;
  last_bytes_recvd = asio::recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data), &total_bytes_recvd);
  UNIT_TEST_CHECK(last_bytes_recvd == sizeof(recv_data) % 10);
  UNIT_TEST_CHECK(total_bytes_recvd == sizeof(recv_data));
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));
}

void async_recv_at_least_n_handler(const asio::error& e,
    size_t last_bytes_recvd, size_t total_bytes_recvd,
    size_t expected_last_bytes_recvd, size_t expected_total_bytes_recvd,
    bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(last_bytes_recvd == expected_last_bytes_recvd);
  UNIT_TEST_CHECK(total_bytes_recvd == expected_total_bytes_recvd);
}

void test_async_recv_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);
  char recv_buf[1024];

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  bool called = false;
  asio::async_recv_at_least_n(s, recv_buf, 1, sizeof(recv_data),
      boost::bind(async_recv_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd,
        sizeof(recv_data), sizeof(recv_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      boost::bind(async_recv_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd,
        sizeof(recv_data), sizeof(recv_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data),
      boost::bind(async_recv_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd,
        sizeof(recv_data), sizeof(recv_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_at_least_n(s, recv_buf, 1, sizeof(recv_data),
      boost::bind(async_recv_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd, 1, 1,
        &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, 1));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      boost::bind(async_recv_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd, 1, 10,
        &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(1);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data),
      boost::bind(async_recv_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd, 1,
        sizeof(recv_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_at_least_n(s, recv_buf, 1, sizeof(recv_data),
      boost::bind(async_recv_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd, 10, 10,
        &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_at_least_n(s, recv_buf, 10, sizeof(recv_data),
      boost::bind(async_recv_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd, 10, 10,
        &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, 10));

  s.reset(recv_data, sizeof(recv_data));
  s.next_recv_length(10);
  memset(recv_buf, 0, sizeof(recv_buf));
  called = false;
  asio::async_recv_at_least_n(s, recv_buf, sizeof(recv_data),
      sizeof(recv_data),
      boost::bind(async_recv_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_recvd, asio::arg::total_bytes_recvd,
        sizeof(recv_data) % 10, sizeof(recv_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(recv_buf, sizeof(recv_data)));
}

void recv_test()
{
  test_recv();
  test_recv_with_error_handler();
  test_async_recv();

  test_recv_n();
  test_recv_n_with_error_handler();
  test_async_recv_n();

  test_recv_at_least_n();
  test_recv_at_least_n_with_error_handler();
  test_async_recv_at_least_n();
}

UNIT_TEST(recv_test)
