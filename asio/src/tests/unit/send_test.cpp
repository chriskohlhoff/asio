//
// send_test.cpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
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
      length_(max_length),
      position_(0),
      next_send_length_(max_length)
  {
    memset(data_, 0, max_length);
  }

  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  void reset(size_t length = max_length)
  {
    UNIT_TEST_CHECK(length <= max_length);

    memset(data_, 0, max_length);
    length_ = length;
    position_ = 0;
    next_send_length_ = length;
  }

  void next_send_length(size_t length)
  {
    next_send_length_ = length;
  }

  bool check(const void* data, size_t length)
  {
    if (length != position_)
      return false;

    return memcmp(data_, data, length) == 0;
  }

  size_t send(const void* data, size_t length)
  {
    if (length > length_ - position_)
      length = length_ - position_;

    if (length > next_send_length_)
      length = next_send_length_;

    memcpy(data_ + position_, data, length);
    position_ += length;

    return length;
  }

  template <typename Error_Handler>
  size_t send(const void* data, size_t length, Error_Handler error_handler)
  {
    if (length > length_ - position_)
      length = length_ - position_;

    if (length > next_send_length_)
      length = next_send_length_;

    memcpy(data_ + position_, data, length);
    position_ += length;

    return length;
  }

  template <typename Handler>
  void async_send(const void* data, size_t length, Handler handler)
  {
    size_t bytes_sent = send(data, length);
    asio::error error;
    demuxer_.post(asio::detail::bind_handler(handler, error, bytes_sent));
  }

private:
  demuxer_type& demuxer_;
  enum { max_length = 8192 };
  char data_[max_length];
  size_t length_;
  size_t position_;
  size_t next_send_length_;
};

static const char send_data[]
  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

void test_send()
{
  asio::demuxer d;
  test_stream s(d);

  s.reset();
  size_t last_bytes_sent = asio::send(s, send_data, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send(s, send_data, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, 1));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send(s, send_data, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));
}

void test_send_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);

  s.reset();
  size_t last_bytes_sent = asio::send(s, send_data, sizeof(send_data),
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send(s, send_data, sizeof(send_data),
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, 1));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send(s, send_data, sizeof(send_data),
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));
}

void async_send_handler(const asio::error& e, size_t bytes_sent,
    size_t expected_bytes_sent, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(bytes_sent == expected_bytes_sent);
}

void test_async_send()
{
  asio::demuxer d;
  test_stream s(d);

  s.reset();
  bool called = false;
  asio::async_send(s, send_data, sizeof(send_data),
      boost::bind(async_send_handler, asio::arg::error, asio::arg::bytes_sent,
        sizeof(send_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  called = false;
  asio::async_send(s, send_data, sizeof(send_data),
      boost::bind(async_send_handler, asio::arg::error, asio::arg::bytes_sent,
        1, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, 1));

  s.reset();
  s.next_send_length(10);
  called = false;
  asio::async_send(s, send_data, sizeof(send_data),
      boost::bind(async_send_handler, asio::arg::error, asio::arg::bytes_sent,
        10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, 10));
}

void test_send_n()
{
  asio::demuxer d;
  test_stream s(d);

  s.reset();
  size_t last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  size_t total_bytes_sent = 0;
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data),
      &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data),
      &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data) % 10);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data),
      &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data) % 10);
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));
}

void test_send_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);

  s.reset();
  size_t last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data), 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  size_t total_bytes_sent = 0;
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data),
      &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data), 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data),
      &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data), 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data) % 10);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_n(s, send_data, sizeof(send_data),
      &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data) % 10);
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));
}

void async_send_n_handler(const asio::error& e, size_t last_bytes_sent,
    size_t total_bytes_sent, size_t expected_last_bytes_sent,
    size_t expected_total_bytes_sent, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(last_bytes_sent == expected_last_bytes_sent);
  UNIT_TEST_CHECK(total_bytes_sent == expected_total_bytes_sent);
}

void test_async_send_n()
{
  asio::demuxer d;
  test_stream s(d);

  s.reset();
  bool called = false;
  asio::async_send_n(s, send_data, sizeof(send_data),
      boost::bind(async_send_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent,
        sizeof(send_data), sizeof(send_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  called = false;
  asio::async_send_n(s, send_data, sizeof(send_data),
      boost::bind(async_send_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent, 1,
        sizeof(send_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  called = false;
  asio::async_send_n(s, send_data, sizeof(send_data),
      boost::bind(async_send_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent,
        sizeof(send_data) % 10, sizeof(send_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));
}

void test_send_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);

  s.reset();
  size_t last_bytes_sent = asio::send_at_least_n(s, send_data, 1,
      sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  size_t total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data),
      &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data),
      &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data), &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, 1));

  s.reset();
  s.next_send_length(1);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data),
      &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(total_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, 1));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(1);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data),
      &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(total_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data), &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data),
      &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(total_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data),
      &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(total_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data));
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data) % 10);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data), &total_bytes_sent);
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data) % 10);
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));
}

void test_send_at_least_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);

  s.reset();
  size_t last_bytes_sent = asio::send_at_least_n(s, send_data, 1,
      sizeof(send_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  size_t total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data),
      &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data),
      0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data),
      &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data), &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data),
      0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, 1));

  s.reset();
  s.next_send_length(1);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data),
      &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(total_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, 1));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data),
      0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(1);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data),
      &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(total_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(1);
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data), &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 1);
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data),
      0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 1, sizeof(send_data),
      &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(total_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data),
      0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, 10, sizeof(send_data),
      &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == 10);
  UNIT_TEST_CHECK(total_bytes_sent == 10);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data) % 10);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  total_bytes_sent = 0;
  last_bytes_sent = asio::send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data), &total_bytes_sent, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_sent == sizeof(send_data) % 10);
  UNIT_TEST_CHECK(total_bytes_sent == sizeof(send_data));
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));
}

void async_send_at_least_n_handler(const asio::error& e,
    size_t last_bytes_sent, size_t total_bytes_sent,
    size_t expected_last_bytes_sent, size_t expected_total_bytes_sent,
    bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(last_bytes_sent == expected_last_bytes_sent);
  UNIT_TEST_CHECK(total_bytes_sent == expected_total_bytes_sent);
}

void test_async_send_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);

  s.reset();
  bool called = false;
  asio::async_send_at_least_n(s, send_data, 1, sizeof(send_data),
      boost::bind(async_send_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent,
        sizeof(send_data), sizeof(send_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  called = false;
  asio::async_send_at_least_n(s, send_data, 10, sizeof(send_data),
      boost::bind(async_send_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent,
        sizeof(send_data), sizeof(send_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  called = false;
  asio::async_send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data),
      boost::bind(async_send_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent,
        sizeof(send_data), sizeof(send_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(1);
  called = false;
  asio::async_send_at_least_n(s, send_data, 1, sizeof(send_data),
      boost::bind(async_send_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent, 1, 1,
        &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, 1));

  s.reset();
  s.next_send_length(1);
  called = false;
  asio::async_send_at_least_n(s, send_data, 10, sizeof(send_data),
      boost::bind(async_send_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent, 1, 10,
        &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(1);
  called = false;
  asio::async_send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data),
      boost::bind(async_send_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent, 1,
        sizeof(send_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));

  s.reset();
  s.next_send_length(10);
  called = false;
  asio::async_send_at_least_n(s, send_data, 1, sizeof(send_data),
      boost::bind(async_send_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent, 10, 10,
        &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  called = false;
  asio::async_send_at_least_n(s, send_data, 10, sizeof(send_data),
      boost::bind(async_send_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent, 10, 10,
        &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, 10));

  s.reset();
  s.next_send_length(10);
  called = false;
  asio::async_send_at_least_n(s, send_data, sizeof(send_data),
      sizeof(send_data),
      boost::bind(async_send_at_least_n_handler, asio::arg::error,
        asio::arg::last_bytes_sent, asio::arg::total_bytes_sent,
        sizeof(send_data) % 10, sizeof(send_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(send_data, sizeof(send_data)));
}

void send_test()
{
  test_send();
  test_send_with_error_handler();
  test_async_send();

  test_send_n();
  test_send_n_with_error_handler();
  test_async_send_n();

  test_send_at_least_n();
  test_send_at_least_n_with_error_handler();
  test_async_send_at_least_n();
}

UNIT_TEST(send_test)
