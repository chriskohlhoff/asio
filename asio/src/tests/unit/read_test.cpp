//
// read_test.cpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/read.hpp"

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
    BOOST_CHECK(length <= max_length);

    memcpy(data_, data, length);
    length_ = length;
    position_ = 0;
    next_read_length_ = length;
  }

  void next_read_length(size_t length)
  {
    next_read_length_ = length;
  }

  template <typename Const_Buffers>
  bool check(const Const_Buffers& buffers, size_t length)
  {
    if (length != position_)
      return false;

    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    size_t checked_length = 0;
    for (; iter != end && checked_length < length; ++iter)
    {
      size_t buffer_length = iter->size();
      if (buffer_length > length - checked_length)
        buffer_length = length - checked_length;
      if (memcmp(data_ + checked_length, iter->data(), buffer_length) != 0)
        return false;
      checked_length += buffer_length;
    }

    return true;
  }

  template <typename Mutable_Buffers>
  size_t read(const Mutable_Buffers& buffers)
  {
    size_t total_length = 0;

    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    for (; iter != end && total_length < next_read_length_; ++iter)
    {
      size_t length = iter->size();
      if (length > length_ - position_)
        length = length_ - position_;

      if (length > next_read_length_ - total_length)
        length = next_read_length_ - total_length;

      memcpy(iter->data(), data_ + position_, length);
      position_ += length;
      total_length += length;
    }

    return total_length;
  }

  template <typename Mutable_Buffers, typename Error_Handler>
  size_t read(const Mutable_Buffers& buffers, Error_Handler)
  {
    return read(buffers);
  }

  template <typename Mutable_Buffers, typename Handler>
  void async_read(const Mutable_Buffers& buffers, Handler handler)
  {
    size_t bytes_transferred = read(buffers);
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
  asio::mutable_buffers<1> buffers = asio::buffers(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read(s, buffers);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read(s, buffers);
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read(s, buffers);
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));
}

void test_read_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];
  asio::mutable_buffers<1> buffers = asio::buffers(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read(s, buffers, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read(s, buffers, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read(s, buffers, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));
}

void async_read_handler(const asio::error& e, size_t bytes_transferred,
    size_t expected_bytes_transferred, bool* called)
{
  *called = true;
  BOOST_CHECK(bytes_transferred == expected_bytes_transferred);
}

void test_async_read()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[1024];
  asio::mutable_buffers<1> buffers = asio::buffers(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read(s, buffers,
      boost::bind(async_read_handler, asio::placeholders::error,
        asio::placeholders::bytes_transferred, sizeof(read_data), &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read(s, buffers,
      boost::bind(async_read_handler, asio::placeholders::error,
        asio::placeholders::bytes_transferred, 1, &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read(s, buffers,
      boost::bind(async_read_handler, asio::placeholders::error,
        asio::placeholders::bytes_transferred, 10, &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, 10));
}

void test_read_n()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers<1> buffers = asio::buffers(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read_n(s, buffers);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, buffers, &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_n(s, buffers);
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, buffers, &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_n(s, buffers);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, buffers, &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));
}

void test_read_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers<1> buffers = asio::buffers(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read_n(s, buffers, 0,
      asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, buffers, &total_bytes_transferred,
      asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_n(s, buffers, 0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, buffers, &total_bytes_transferred,
      asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_n(s, buffers, 0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_n(s, buffers, &total_bytes_transferred,
      asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));
}

void async_read_n_handler(const asio::error& e, size_t last_bytes_transferred,
    size_t total_bytes_transferred, size_t expected_last_bytes_transferred,
    size_t expected_total_bytes_transferred, bool* called)
{
  *called = true;
  BOOST_CHECK(last_bytes_transferred == expected_last_bytes_transferred);
  BOOST_CHECK(total_bytes_transferred == expected_total_bytes_transferred);
}

void test_async_read_n()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers<1> buffers = asio::buffers(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read_n(s, buffers,
      boost::bind(async_read_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data), sizeof(read_data), &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_n(s, buffers,
      boost::bind(async_read_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 1,
        sizeof(read_data), &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_n(s, buffers,
      boost::bind(async_read_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data) % 10, sizeof(read_data), &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));
}

void test_read_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers<1> buffers = asio::buffers(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read_at_least_n(s, buffers, 1);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1,
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10,
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data));
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data),
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1);
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1,
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(total_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10);
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10,
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(total_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data));
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data),
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1);
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1,
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(total_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10);
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10,
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(total_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data));
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data),
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));
}

void test_read_at_least_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers<1> buffers = asio::buffers(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t last_bytes_transferred = asio::read_at_least_n(s, buffers, 1,
      0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1,
      &total_bytes_transferred, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10,
      0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10,
      &total_bytes_transferred, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data),
      0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data),
      &total_bytes_transferred, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1,
      0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1,
      &total_bytes_transferred, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(total_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10,
      0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10,
      &total_bytes_transferred, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(total_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data),
      0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data),
      &total_bytes_transferred, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 1);
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1,
      0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 1,
      &total_bytes_transferred, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(total_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10,
      0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, 10,
      &total_bytes_transferred, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == 10);
  BOOST_CHECK(total_bytes_transferred == 10);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data),
      0, asio::ignore_error());
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::read_at_least_n(s, buffers, sizeof(read_data),
      &total_bytes_transferred);
  BOOST_CHECK(last_bytes_transferred == sizeof(read_data) % 10);
  BOOST_CHECK(total_bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));
}

void async_read_at_least_n_handler(const asio::error& e,
    size_t last_bytes_transferred, size_t total_bytes_transferred,
    size_t expected_last_bytes_transferred,
    size_t expected_total_bytes_transferred, bool* called)
{
  *called = true;
  BOOST_CHECK(last_bytes_transferred == expected_last_bytes_transferred);
  BOOST_CHECK(total_bytes_transferred == expected_total_bytes_transferred);
}

void test_async_read_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers<1> buffers = asio::buffers(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read_at_least_n(s, buffers, 1,
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data), sizeof(read_data), &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, buffers, 10,
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data), sizeof(read_data), &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, buffers, sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data), sizeof(read_data), &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, buffers, 1,
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 1, 1, &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, buffers, 10,
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 1, 10, &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, buffers, sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        1, sizeof(read_data), &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, buffers, 1,
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 10, 10, &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, buffers, 10,
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 10, 10, &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at_least_n(s, buffers, sizeof(read_data),
      boost::bind(async_read_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(read_data) % 10, sizeof(read_data), &called));
  d.reset();
  d.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check(buffers, sizeof(read_data)));
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("read");
  test->add(BOOST_TEST_CASE(&test_read));
  test->add(BOOST_TEST_CASE(&test_read_with_error_handler));
  test->add(BOOST_TEST_CASE(&test_async_read));
  test->add(BOOST_TEST_CASE(&test_read_n));
  test->add(BOOST_TEST_CASE(&test_read_n_with_error_handler));
  test->add(BOOST_TEST_CASE(&test_async_read_n));
  test->add(BOOST_TEST_CASE(&test_read_at_least_n));
  test->add(BOOST_TEST_CASE(&test_read_at_least_n_with_error_handler));
  test->add(BOOST_TEST_CASE(&test_async_read_at_least_n));
  return test;
}
