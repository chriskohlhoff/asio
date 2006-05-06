//
// is_write_buffered_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/is_write_buffered.hpp"

#include <boost/bind.hpp>
#include <boost/noncopyable.hpp>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace std; // For memcmp, memcpy and memset.

class test_stream
  : private boost::noncopyable
{
public:
  typedef asio::io_service io_service_type;

  typedef asio::error error_type;

  typedef test_stream lowest_layer_type;

  test_stream(asio::io_service& io_service)
    : io_service_(io_service)
  {
  }

  io_service_type& io_service()
  {
    return io_service_;
  }

  lowest_layer_type& lowest_layer()
  {
    return *this;
  }

  template <typename Const_Buffers>
  size_t write(const Const_Buffers&)
  {
    return 0;
  }

  template <typename Const_Buffers, typename Error_Handler>
  size_t write(const Const_Buffers&, Error_Handler)
  {
    return 0;
  }

  template <typename Const_Buffers, typename Handler>
  void async_write(const Const_Buffers&, Handler handler)
  {
    asio::error error;
    io_service_.post(asio::detail::bind_handler(handler, error, 0));
  }

  template <typename Mutable_Buffers>
  size_t read(const Mutable_Buffers&)
  {
    return 0;
  }

  template <typename Mutable_Buffers, typename Error_Handler>
  size_t read(const Mutable_Buffers&, Error_Handler error_handler)
  {
    return 0;
  }

  template <typename Mutable_Buffers, typename Handler>
  void async_read(const Mutable_Buffers&, Handler handler)
  {
    asio::error error;
    io_service_.post(asio::detail::bind_handler(handler, error, 0));
  }

private:
  io_service_type& io_service_;
};

void is_write_buffered_test()
{
  BOOST_CHECK(!asio::is_write_buffered<
      asio::ip::tcp::socket>::value);

  BOOST_CHECK(!asio::is_write_buffered<
      asio::buffered_read_stream<
        asio::ip::tcp::socket> >::value);

  BOOST_CHECK(!!asio::is_write_buffered<
      asio::buffered_write_stream<
        asio::ip::tcp::socket> >::value);

  BOOST_CHECK(!!asio::is_write_buffered<
      asio::buffered_stream<asio::ip::tcp::socket> >::value);

  BOOST_CHECK(!asio::is_write_buffered<test_stream>::value);

  BOOST_CHECK(!asio::is_write_buffered<
      asio::buffered_read_stream<test_stream> >::value);

  BOOST_CHECK(!!asio::is_write_buffered<
      asio::buffered_write_stream<test_stream> >::value);

  BOOST_CHECK(!!asio::is_write_buffered<
      asio::buffered_stream<test_stream> >::value);
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("is_write_buffered");
  test->add(BOOST_TEST_CASE(&is_write_buffered_test));
  return test;
}
