//
// read_at.cpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/read_at.hpp"

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/noncopyable.hpp>
#include <cstring>
#include "asio/io_service.hpp"
#include "asio/placeholders.hpp"
#include "asio/streambuf.hpp"
#include "unit_test.hpp"

using namespace std; // For memcmp, memcpy and memset.

class test_random_access_device
  : private boost::noncopyable
{
public:
  typedef asio::io_service io_service_type;

  test_random_access_device(asio::io_service& io_service)
    : io_service_(io_service),
      length_(0),
      next_read_length_(0)
  {
  }

  io_service_type& get_io_service()
  {
    return io_service_;
  }

  void reset(const void* data, size_t length)
  {
    BOOST_CHECK(length <= max_length);

    length_ = 0;
    while (length_ + length < max_length)
    {
      memcpy(data_ + length_, data, length);
      length_ += length;
    }

    next_read_length_ = length;
  }

  void next_read_length(size_t length)
  {
    next_read_length_ = length;
  }

  template <typename Const_Buffers>
  bool check_buffers(boost::uint64_t offset,
      const Const_Buffers& buffers, size_t length)
  {
    if (offset + length > max_length)
      return false;

    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    size_t checked_length = 0;
    for (; iter != end && checked_length < length; ++iter)
    {
      size_t buffer_length = asio::buffer_size(*iter);
      if (buffer_length > length - checked_length)
        buffer_length = length - checked_length;
      if (memcmp(data_ + offset + checked_length,
            asio::buffer_cast<const void*>(*iter), buffer_length) != 0)
        return false;
      checked_length += buffer_length;
    }

    return true;
  }

  template <typename Mutable_Buffers>
  size_t read_some_at(boost::uint64_t offset, const Mutable_Buffers& buffers)
  {
    size_t total_length = 0;

    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    for (; iter != end && total_length < next_read_length_; ++iter)
    {
      size_t length = asio::buffer_size(*iter);
      if (length > length_ - offset)
        length = length_ - offset;

      if (length > next_read_length_ - total_length)
        length = next_read_length_ - total_length;

      memcpy(asio::buffer_cast<void*>(*iter),
          data_ + offset, length);
      offset += length;
      total_length += length;
    }

    return total_length;
  }

  template <typename Mutable_Buffers>
  size_t read_some_at(boost::uint64_t offset,
      const Mutable_Buffers& buffers, asio::error_code& ec)
  {
    ec = asio::error_code();
    return read_some_at(offset, buffers);
  }

  template <typename Mutable_Buffers, typename Handler>
  void async_read_some_at(boost::uint64_t offset,
      const Mutable_Buffers& buffers, Handler handler)
  {
    size_t bytes_transferred = read_some_at(offset, buffers);
    io_service_.post(asio::detail::bind_handler(
          handler, asio::error_code(), bytes_transferred));
  }

private:
  io_service_type& io_service_;
  enum { max_length = 8192 };
  char data_[max_length];
  size_t length_;
  size_t next_read_length_;
};

static const char read_data[]
  = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

void test_3_arg_mutable_buffers_1_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers_1 buffers
    = asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t bytes_transferred = asio::read_at(s, 0, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
}

void test_3_arg_multi_buffers_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  boost::array<asio::mutable_buffer, 2> buffers = { {
    asio::buffer(read_buf, 32),
    asio::buffer(read_buf) + 32 } };

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t bytes_transferred = asio::read_at(s, 0, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
}

void test_3_arg_streambuf_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  size_t bytes_transferred = asio::read_at(s, 0, sb);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
}

bool old_style_transfer_all(const asio::error_code& ec,
    size_t /*bytes_transferred*/)
{
  return !!ec;
}

size_t short_transfer(const asio::error_code& ec,
    size_t /*bytes_transferred*/)
{
  return !!ec ? 0 : 3;
}

void test_4_arg_mutable_buffers_1_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers_1 buffers
    = asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
}

void test_4_arg_multi_buffers_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  boost::array<asio::mutable_buffer, 2> buffers = { {
    asio::buffer(read_buf, 32),
    asio::buffer(read_buf) + 32 } };

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  size_t bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
}

void test_4_arg_streambuf_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  size_t bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_all());
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(sb.size() == 1);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(sb.size() == 1);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(1));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(10));
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(sb.size() == 42);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(sb.size() == 42);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(sb.size() == 50);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(42));
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(sb.size() == 50);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      old_style_transfer_all);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb, short_transfer);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
}

void test_5_arg_mutable_buffers_1_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers_1 buffers
    = asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  asio::error_code error;
  size_t bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(s.check_buffers(0, buffers, 1));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(s.check_buffers(1234, buffers, 1));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(s.check_buffers(0, buffers, 42));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(s.check_buffers(1234, buffers, 42));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(s.check_buffers(0, buffers, 50));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(s.check_buffers(1234, buffers, 50));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);
}

void test_5_arg_multi_buffers_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  boost::array<asio::mutable_buffer, 2> buffers = { {
    asio::buffer(read_buf, 32),
    asio::buffer(read_buf) + 32 } };

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  asio::error_code error;
  size_t bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(s.check_buffers(0, buffers, 1));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(s.check_buffers(1234, buffers, 1));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(s.check_buffers(0, buffers, 42));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(s.check_buffers(1234, buffers, 42));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(s.check_buffers(0, buffers, 50));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(s.check_buffers(1234, buffers, 50));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 0, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bytes_transferred = asio::read_at(s, 1234, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, buffers,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
  BOOST_CHECK(!error);
}

void test_5_arg_streambuf_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  asio::error_code error;
  size_t bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_all(), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(sb.size() == 1);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 1));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 1);
  BOOST_CHECK(sb.size() == 1);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 1));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(1), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(10), error);
  BOOST_CHECK(bytes_transferred == 10);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 10));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(sb.size() == 42);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 42));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 42);
  BOOST_CHECK(sb.size() == 42);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 42));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(sb.size() == 50);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 50));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      asio::transfer_at_least(42), error);
  BOOST_CHECK(bytes_transferred == 50);
  BOOST_CHECK(sb.size() == 50);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 50));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      old_style_transfer_all, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 0, sb,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bytes_transferred = asio::read_at(s, 1234, sb,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 0, sb,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  error = asio::error_code();
  bytes_transferred = asio::read_at(s, 1234, sb,
      short_transfer, error);
  BOOST_CHECK(bytes_transferred == sizeof(read_data));
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
  BOOST_CHECK(!error);
}

void async_read_handler(const asio::error_code& e,
    size_t bytes_transferred, size_t expected_bytes_transferred, bool* called)
{
  *called = true;
  BOOST_CHECK(!e);
  BOOST_CHECK(bytes_transferred == expected_bytes_transferred);
}

void test_4_arg_mutable_buffers_1_async_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers_1 buffers
    = asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read_at(s, 0, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
}

void test_4_arg_multi_buffers_async_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  boost::array<asio::mutable_buffer, 2> buffers = { {
    asio::buffer(read_buf, 32),
    asio::buffer(read_buf) + 32 } };

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read_at(s, 0, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
}

void test_4_arg_streambuf_async_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bool called = false;
  asio::async_read_at(s, 0, sb,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
}

void test_5_arg_mutable_buffers_1_async_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  asio::mutable_buffers_1 buffers
    = asio::buffer(read_buf, sizeof(read_buf));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        1, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        1, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        42, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        42, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        50, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        50, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
}

void test_5_arg_multi_buffers_async_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  char read_buf[sizeof(read_data)];
  boost::array<asio::mutable_buffer, 2> buffers = { {
    asio::buffer(read_buf, 32),
    asio::buffer(read_buf) + 32 } };

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  bool called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        1, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        1, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 10));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        42, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        42, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        50, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        50, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, 50));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 0, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(0, buffers, sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  memset(read_buf, 0, sizeof(read_buf));
  called = false;
  asio::async_read_at(s, 1234, buffers, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(s.check_buffers(1234, buffers, sizeof(read_data)));
}

void test_5_arg_streambuf_async_read_at()
{
  asio::io_service ios;
  test_random_access_device s(ios);
  asio::streambuf sb(sizeof(read_data));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  bool called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_all(),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        1, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 1);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        1, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 1);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 1));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_at_least(1),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_at_least(10),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        10, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 10);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 10));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        42, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 42);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        42, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 42);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 42));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        50, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 50);
  BOOST_CHECK(s.check_buffers(0, sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb,
      asio::transfer_at_least(42),
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        50, &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == 50);
  BOOST_CHECK(s.check_buffers(1234, sb.data(), 50));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb, old_style_transfer_all,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(1);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 0, sb, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(0, sb.data(), sizeof(read_data)));

  s.reset(read_data, sizeof(read_data));
  s.next_read_length(10);
  sb.consume(sb.size());
  called = false;
  asio::async_read_at(s, 1234, sb, short_transfer,
      boost::bind(async_read_handler,
        asio::placeholders::error,
        asio::placeholders::bytes_transferred,
        sizeof(read_data), &called));
  ios.reset();
  ios.run();
  BOOST_CHECK(called);
  BOOST_CHECK(sb.size() == sizeof(read_data));
  BOOST_CHECK(s.check_buffers(1234, sb.data(), sizeof(read_data)));
}

test_suite* init_unit_test_suite(int, char*[])
{
  test_suite* test = BOOST_TEST_SUITE("read_at");
  test->add(BOOST_TEST_CASE(&test_3_arg_mutable_buffers_1_read_at));
  test->add(BOOST_TEST_CASE(&test_3_arg_multi_buffers_read_at));
  test->add(BOOST_TEST_CASE(&test_3_arg_streambuf_read_at));
  test->add(BOOST_TEST_CASE(&test_4_arg_mutable_buffers_1_read_at));
  test->add(BOOST_TEST_CASE(&test_4_arg_multi_buffers_read_at));
  test->add(BOOST_TEST_CASE(&test_4_arg_streambuf_read_at));
  test->add(BOOST_TEST_CASE(&test_5_arg_mutable_buffers_1_read_at));
  test->add(BOOST_TEST_CASE(&test_5_arg_multi_buffers_read_at));
  test->add(BOOST_TEST_CASE(&test_5_arg_streambuf_read_at));
  test->add(BOOST_TEST_CASE(&test_4_arg_mutable_buffers_1_async_read_at));
  test->add(BOOST_TEST_CASE(&test_4_arg_multi_buffers_async_read_at));
  test->add(BOOST_TEST_CASE(&test_4_arg_streambuf_async_read_at));
  test->add(BOOST_TEST_CASE(&test_5_arg_mutable_buffers_1_async_read_at));
  test->add(BOOST_TEST_CASE(&test_5_arg_multi_buffers_async_read_at));
  test->add(BOOST_TEST_CASE(&test_5_arg_streambuf_async_read_at));
  return test;
}
