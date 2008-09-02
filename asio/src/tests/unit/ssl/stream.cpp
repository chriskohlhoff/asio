//
// stream.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/ssl/stream.hpp"

#include "asio.hpp"
#include "asio/ssl.hpp"
#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// ssl_stream_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ssl::stream::socket compile and link correctly. Runtime failures are ignored.

namespace ssl_stream_compile {

void handshake_handler(const asio::error_code&)
{
}

void shutdown_handler(const asio::error_code&)
{
}

void write_some_handler(const asio::error_code&, std::size_t)
{
}

void read_some_handler(const asio::error_code&, std::size_t)
{
}

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  try
  {
    io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    asio::ssl::context context(ios, asio::ssl::context::sslv23);
    asio::error_code ec;

    // ssl::stream constructors.

    ssl::stream<ip::tcp::socket> stream1(ios, context);
    ip::tcp::socket socket1(ios, ip::tcp::v4());
    ssl::stream<ip::tcp::socket&> stream2(socket1, context);

    // basic_io_object functions.

    io_service& ios_ref = stream1.io_service();
    (void)ios_ref;

    // ssl::stream functions.

    ssl::stream<ip::tcp::socket>::lowest_layer_type& lowest_layer
      = stream1.lowest_layer();
    (void)lowest_layer;

    const ssl::stream<ip::tcp::socket>& stream3 = stream1;
    const ssl::stream<ip::tcp::socket>::lowest_layer_type& lowest_layer2
      = stream3.lowest_layer();
    (void)lowest_layer2;

    stream1.handshake(ssl::stream_base::client);
    stream1.handshake(ssl::stream_base::server);
    stream1.handshake(ssl::stream_base::client, ec);
    stream1.handshake(ssl::stream_base::server, ec);

    stream1.async_handshake(ssl::stream_base::client, handshake_handler);
    stream1.async_handshake(ssl::stream_base::server, handshake_handler);

    stream1.shutdown();
    stream1.shutdown(ec);

    stream1.async_shutdown(shutdown_handler);

    stream1.write_some(buffer(mutable_char_buffer));
    stream1.write_some(buffer(const_char_buffer));
    stream1.write_some(buffer(mutable_char_buffer), ec);
    stream1.write_some(buffer(const_char_buffer), ec);

    stream1.async_write_some(buffer(mutable_char_buffer), write_some_handler);
    stream1.async_write_some(buffer(const_char_buffer), write_some_handler);

    stream1.read_some(buffer(mutable_char_buffer));
    stream1.read_some(buffer(mutable_char_buffer), ec);

    stream1.async_read_some(buffer(mutable_char_buffer), read_some_handler);

    stream1.peek(buffer(mutable_char_buffer));
    stream1.peek(buffer(mutable_char_buffer), ec);

    std::size_t in_avail1 = stream1.in_avail();
    (void)in_avail1;
    std::size_t in_avail2 = stream1.in_avail(ec);
    (void)in_avail2;
  }
  catch (std::exception&)
  {
  }
}

} // namespace ssl_stream_compile

//------------------------------------------------------------------------------

test_suite* init_unit_test_suite(int, char*[])
{
  test_suite* test = BOOST_TEST_SUITE("ssl/stream");
  test->add(BOOST_TEST_CASE(&ssl_stream_compile::test));
  return test;
}
