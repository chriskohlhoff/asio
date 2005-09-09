//
// write_test.cpp
// ~~~~~~~~~~~~~~
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
      length_(max_length),
      position_(0),
      next_write_length_(max_length)
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
    next_write_length_ = length;
  }

  void next_write_length(size_t length)
  {
    next_write_length_ = length;
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

  template <typename Const_Buffers>
  size_t write(const Const_Buffers& buffers)
  {
    size_t total_length = 0;

    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    for (; iter != end && total_length < next_write_length_; ++iter)
    {
      size_t length = iter->size();
      if (length > length_ - position_)
        length = length_ - position_;

      if (length > next_write_length_ - total_length)
        length = next_write_length_ - total_length;

      memcpy(data_ + position_, iter->data(), length);
      position_ += length;
      total_length += length;
    }

    return total_length;
  }

  template <typename Const_Buffers, typename Error_Handler>
  size_t write(const Const_Buffers& buffers, Error_Handler)
  {
    return write(buffers);
  }

  template <typename Const_Buffers, typename Handler>
  void async_write(const Const_Buffers& buffers, Handler handler)
  {
    size_t bytes_transferred = write(buffers);
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
  size_t next_write_length_;
};

static const char write_data[]
  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

void test_write()
{
  asio::demuxer d;
  test_stream s(d);
  asio::const_buffers<1> buffers =
    asio::buffers(write_data, sizeof(write_data));

  s.reset();
  size_t last_bytes_transferred = asio::write(s, buffers);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write(s, buffers);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, 1));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write(s, buffers);
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));
}

void test_write_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  asio::const_buffers<1> buffers =
    asio::buffers(write_data, sizeof(write_data));

  s.reset();
  size_t last_bytes_transferred = asio::write(s, buffers, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write(s, buffers, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, 1));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write(s, buffers, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));
}

void async_write_handler(const asio::error& e, size_t bytes_transferred,
    size_t expected_bytes_transferred, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(bytes_transferred == expected_bytes_transferred);
}

void test_async_write()
{
  asio::demuxer d;
  test_stream s(d);
  asio::const_buffers<1> buffers =
    asio::buffers(write_data, sizeof(write_data));

  s.reset();
  bool called = false;
  asio::async_write(s, buffers,
      boost::bind(async_write_handler, asio::placeholders::error,
        asio::placeholders::bytes_transferred, sizeof(write_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  called = false;
  asio::async_write(s, buffers,
      boost::bind(async_write_handler, asio::placeholders::error,
        asio::placeholders::bytes_transferred, 1, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, 1));

  s.reset();
  s.next_write_length(10);
  called = false;
  asio::async_write(s, buffers,
      boost::bind(async_write_handler, asio::placeholders::error,
        asio::placeholders::bytes_transferred, 10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, 10));
}

void test_write_n()
{
  asio::demuxer d;
  test_stream s(d);
  asio::const_buffers<1> buffers =
    asio::buffers(write_data, sizeof(write_data));

  s.reset();
  size_t last_bytes_transferred = asio::write_n(s, buffers);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_n(s, buffers, &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write_n(s, buffers);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_n(s, buffers, &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write_n(s, buffers);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data) % 10);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_n(s, buffers, &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data) % 10);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));
}

void test_write_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  asio::const_buffers<1> buffers =
    asio::buffers(write_data, sizeof(write_data));

  s.reset();
  size_t last_bytes_transferred = asio::write_n(s, buffers, 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_n(s, buffers, &total_bytes_transferred,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write_n(s, buffers, 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_n(s, buffers, &total_bytes_transferred,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write_n(s, buffers, 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data) % 10);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_n(s, buffers, &total_bytes_transferred,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data) % 10);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));
}

void async_write_n_handler(const asio::error& e, size_t last_bytes_transferred,
    size_t total_bytes_transferred, size_t expected_last_bytes_transferred,
    size_t expected_total_bytes_transferred, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(last_bytes_transferred == expected_last_bytes_transferred);
  UNIT_TEST_CHECK(total_bytes_transferred == expected_total_bytes_transferred);
}

void test_async_write_n()
{
  asio::demuxer d;
  test_stream s(d);
  asio::const_buffers<1> buffers =
    asio::buffers(write_data, sizeof(write_data));

  s.reset();
  bool called = false;
  asio::async_write_n(s, buffers,
      boost::bind(async_write_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(write_data), sizeof(write_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  called = false;
  asio::async_write_n(s, buffers,
      boost::bind(async_write_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        1, sizeof(write_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  called = false;
  asio::async_write_n(s, buffers,
      boost::bind(async_write_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(write_data) % 10, sizeof(write_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));
}

void test_write_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);
  asio::const_buffers<1> buffers =
    asio::buffers(write_data, sizeof(write_data));

  s.reset();
  size_t last_bytes_transferred = asio::write_at_least_n(s, buffers, 1);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1,
      &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10,
      &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, 1));

  s.reset();
  s.next_write_length(1);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1,
      &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, 1));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(1);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10,
      &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data));
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1);
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1,
      &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10);
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10,
      &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data));
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data) % 10);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data), &total_bytes_transferred);
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data) % 10);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));
}

void test_write_at_least_n_with_error_handler()
{
  asio::demuxer d;
  test_stream s(d);
  asio::const_buffers<1> buffers =
    asio::buffers(write_data, sizeof(write_data));

  s.reset();
  size_t last_bytes_transferred = asio::write_at_least_n(s, buffers, 1, 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  size_t total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1,
      &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10, 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10,
      &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data), &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1, 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, 1));

  s.reset();
  s.next_write_length(1);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1,
      &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, 1));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10, 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(1);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10,
      &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(1);
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data), &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 1);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1, 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 1,
      &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10, 0,
      asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers, 10,
      &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == 10);
  UNIT_TEST_CHECK(total_bytes_transferred == 10);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data), 0, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data) % 10);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  total_bytes_transferred = 0;
  last_bytes_transferred = asio::write_at_least_n(s, buffers,
      sizeof(write_data), &total_bytes_transferred, asio::ignore_error());
  UNIT_TEST_CHECK(last_bytes_transferred == sizeof(write_data) % 10);
  UNIT_TEST_CHECK(total_bytes_transferred == sizeof(write_data));
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));
}

void async_write_at_least_n_handler(const asio::error& e,
    size_t last_bytes_transferred, size_t total_bytes_transferred,
    size_t expected_last_bytes_transferred,
    size_t expected_total_bytes_transferred, bool* called)
{
  *called = true;
  UNIT_TEST_CHECK(last_bytes_transferred == expected_last_bytes_transferred);
  UNIT_TEST_CHECK(total_bytes_transferred == expected_total_bytes_transferred);
}

void test_async_write_at_least_n()
{
  asio::demuxer d;
  test_stream s(d);
  asio::const_buffers<1> buffers =
    asio::buffers(write_data, sizeof(write_data));

  s.reset();
  bool called = false;
  asio::async_write_at_least_n(s, buffers, 1,
      boost::bind(async_write_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(write_data), sizeof(write_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  called = false;
  asio::async_write_at_least_n(s, buffers, 10,
      boost::bind(async_write_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(write_data), sizeof(write_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  called = false;
  asio::async_write_at_least_n(s, buffers, sizeof(write_data),
      boost::bind(async_write_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(write_data), sizeof(write_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(1);
  called = false;
  asio::async_write_at_least_n(s, buffers, 1,
      boost::bind(async_write_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 1, 1, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, 1));

  s.reset();
  s.next_write_length(1);
  called = false;
  asio::async_write_at_least_n(s, buffers, 10,
      boost::bind(async_write_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 1, 10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(1);
  called = false;
  asio::async_write_at_least_n(s, buffers, sizeof(write_data),
      boost::bind(async_write_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        1, sizeof(write_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));

  s.reset();
  s.next_write_length(10);
  called = false;
  asio::async_write_at_least_n(s, buffers, 1,
      boost::bind(async_write_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 10, 10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  called = false;
  asio::async_write_at_least_n(s, buffers, 10,
      boost::bind(async_write_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred, 10, 10, &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, 10));

  s.reset();
  s.next_write_length(10);
  called = false;
  asio::async_write_at_least_n(s, buffers, sizeof(write_data),
      boost::bind(async_write_at_least_n_handler, asio::placeholders::error,
        asio::placeholders::last_bytes_transferred,
        asio::placeholders::total_bytes_transferred,
        sizeof(write_data) % 10, sizeof(write_data), &called));
  d.reset();
  d.run();
  UNIT_TEST_CHECK(called);
  UNIT_TEST_CHECK(s.check(buffers, sizeof(write_data)));
}

void write_test()
{
  test_write();
  test_write_with_error_handler();
  test_async_write();

  test_write_n();
  test_write_n_with_error_handler();
  test_async_write_n();

  test_write_at_least_n();
  test_write_at_least_n_with_error_handler();
  test_async_write_at_least_n();
}

UNIT_TEST(write_test)
