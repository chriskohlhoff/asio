//
// is_recv_buffered_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
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

#include <boost/bind.hpp>
#include <boost/noncopyable.hpp>
#include "asio.hpp"
#include "unit_test.hpp"

using namespace std; // For memcmp, memcpy and memset.

class test_stream
  : private boost::noncopyable
{
public:
  typedef asio::demuxer demuxer_type;

  typedef test_stream lowest_layer_type;

  test_stream(asio::demuxer& d)
    : demuxer_(d)
  {
  }

  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  lowest_layer_type& lowest_layer()
  {
    return *this;
  }

  size_t send(const void* data, size_t length)
  {
    return 0;
  }

  template <typename Error_Handler>
  size_t send(const void* data, size_t length, Error_Handler error_handler)
  {
    return 0;
  }

  template <typename Handler>
  void async_send(const void* data, size_t length, Handler handler)
  {
    asio::socket_error error;
    demuxer_.post(asio::detail::bind_handler(handler, error, 0));
  }

  size_t recv(void* data, size_t length)
  {
    return 0;
  }

  template <typename Error_Handler>
  size_t recv(void* data, size_t length, Error_Handler error_handler)
  {
    return 0;
  }

  template <typename Handler>
  void async_recv(void* data, size_t length, Handler handler)
  {
    asio::socket_error error;
    demuxer_.post(asio::detail::bind_handler(handler, error, 0));
  }

private:
  demuxer_type& demuxer_;
};

void is_recv_buffered_test()
{
  UNIT_TEST_CHECK(!asio::is_recv_buffered<asio::stream_socket>::value);

  UNIT_TEST_CHECK(asio::is_recv_buffered<
      asio::buffered_recv_stream<asio::stream_socket> >::value);

  UNIT_TEST_CHECK(!asio::is_recv_buffered<
      asio::buffered_send_stream<asio::stream_socket> >::value);

  UNIT_TEST_CHECK(asio::is_recv_buffered<
      asio::buffered_stream<asio::stream_socket> >::value);

  UNIT_TEST_CHECK(!asio::is_recv_buffered<test_stream>::value);

  UNIT_TEST_CHECK(asio::is_recv_buffered<
      asio::buffered_recv_stream<test_stream> >::value);

  UNIT_TEST_CHECK(!asio::is_recv_buffered<
      asio::buffered_send_stream<test_stream> >::value);

  UNIT_TEST_CHECK(asio::is_recv_buffered<
      asio::buffered_stream<test_stream> >::value);
}

UNIT_TEST(is_recv_buffered_test)
