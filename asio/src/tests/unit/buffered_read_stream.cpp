//
// buffered_read_stream.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2012 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/buffered_read_stream.hpp"

#include <cstring>
#include "asio/buffer.hpp"
#include "asio/io_service.hpp"
#include "asio/ip/tcp.hpp"
#include "asio/system_error.hpp"
#include "unit_test.hpp"

#if defined(ASIO_HAS_BOOST_BIND)
# include <boost/bind.hpp>
#else // defined(ASIO_HAS_BOOST_BIND)
# include <functional>
#endif // defined(ASIO_HAS_BOOST_BIND)

typedef asio::buffered_read_stream<
    asio::ip::tcp::socket> stream_type;

void test_sync_operations()
{
  using namespace std; // For memcmp.

  asio::io_service io_service;

  asio::ip::tcp::acceptor acceptor(io_service,
      asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 0));
  asio::ip::tcp::endpoint server_endpoint = acceptor.local_endpoint();
  server_endpoint.address(asio::ip::address_v4::loopback());

  stream_type client_socket(io_service);
  client_socket.lowest_layer().connect(server_endpoint);

  stream_type server_socket(io_service);
  acceptor.accept(server_socket.lowest_layer());

  const char write_data[]
    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const asio::const_buffer write_buf = asio::buffer(write_data);

  std::size_t bytes_written = 0;
  while (bytes_written < sizeof(write_data))
  {
    bytes_written += client_socket.write_some(
        asio::buffer(write_buf + bytes_written));
  }

  char read_data[sizeof(write_data)];
  const asio::mutable_buffer read_buf = asio::buffer(read_data);

  std::size_t bytes_read = 0;
  while (bytes_read < sizeof(read_data))
  {
    bytes_read += server_socket.read_some(
        asio::buffer(read_buf + bytes_read));
  }

  ASIO_CHECK(bytes_written == sizeof(write_data));
  ASIO_CHECK(bytes_read == sizeof(read_data));
  ASIO_CHECK(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  bytes_written = 0;
  while (bytes_written < sizeof(write_data))
  {
    bytes_written += server_socket.write_some(
        asio::buffer(write_buf + bytes_written));
  }

  bytes_read = 0;
  while (bytes_read < sizeof(read_data))
  {
    bytes_read += client_socket.read_some(
        asio::buffer(read_buf + bytes_read));
  }

  ASIO_CHECK(bytes_written == sizeof(write_data));
  ASIO_CHECK(bytes_read == sizeof(read_data));
  ASIO_CHECK(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  server_socket.close();
  asio::error_code error;
  bytes_read = client_socket.read_some(
      asio::buffer(read_buf), error);

  ASIO_CHECK(bytes_read == 0);
  ASIO_CHECK(error == asio::error::eof);

  client_socket.close(error);
}

void handle_accept(const asio::error_code& e)
{
  ASIO_CHECK(!e);
}

void handle_write(const asio::error_code& e,
    std::size_t bytes_transferred,
    std::size_t* total_bytes_written)
{
  ASIO_CHECK(!e);
  if (e)
    throw asio::system_error(e); // Terminate test.
  *total_bytes_written += bytes_transferred;
}

void handle_read(const asio::error_code& e,
    std::size_t bytes_transferred,
    std::size_t* total_bytes_read)
{
  ASIO_CHECK(!e);
  if (e)
    throw asio::system_error(e); // Terminate test.
  *total_bytes_read += bytes_transferred;
}

void handle_read_eof(const asio::error_code& e,
    std::size_t bytes_transferred)
{
  ASIO_CHECK(e == asio::error::eof);
  ASIO_CHECK(bytes_transferred == 0);
}

void test_async_operations()
{
  using namespace std; // For memcmp.

#if defined(ASIO_HAS_BOOST_BIND)
  namespace bindns = boost;
#else // defined(ASIO_HAS_BOOST_BIND)
  namespace bindns = std;
  using std::placeholders::_1;
  using std::placeholders::_2;
#endif // defined(ASIO_HAS_BOOST_BIND)

  asio::io_service io_service;

  asio::ip::tcp::acceptor acceptor(io_service,
      asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 0));
  asio::ip::tcp::endpoint server_endpoint = acceptor.local_endpoint();
  server_endpoint.address(asio::ip::address_v4::loopback());

  stream_type client_socket(io_service);
  client_socket.lowest_layer().connect(server_endpoint);

  stream_type server_socket(io_service);
  acceptor.async_accept(server_socket.lowest_layer(), &handle_accept);
  io_service.run();
  io_service.reset();

  const char write_data[]
    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const asio::const_buffer write_buf = asio::buffer(write_data);

  std::size_t bytes_written = 0;
  while (bytes_written < sizeof(write_data))
  {
    client_socket.async_write_some(
        asio::buffer(write_buf + bytes_written),
        bindns::bind(handle_write, _1, _2, &bytes_written));
    io_service.run();
    io_service.reset();
  }

  char read_data[sizeof(write_data)];
  const asio::mutable_buffer read_buf = asio::buffer(read_data);

  std::size_t bytes_read = 0;
  while (bytes_read < sizeof(read_data))
  {
    server_socket.async_read_some(
        asio::buffer(read_buf + bytes_read),
        bindns::bind(handle_read, _1, _2, &bytes_read));
    io_service.run();
    io_service.reset();
  }

  ASIO_CHECK(bytes_written == sizeof(write_data));
  ASIO_CHECK(bytes_read == sizeof(read_data));
  ASIO_CHECK(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  bytes_written = 0;
  while (bytes_written < sizeof(write_data))
  {
    server_socket.async_write_some(
        asio::buffer(write_buf + bytes_written),
        bindns::bind(handle_write, _1, _2, &bytes_written));
    io_service.run();
    io_service.reset();
  }

  bytes_read = 0;
  while (bytes_read < sizeof(read_data))
  {
    client_socket.async_read_some(
        asio::buffer(read_buf + bytes_read),
        bindns::bind(handle_read, _1, _2, &bytes_read));
    io_service.run();
    io_service.reset();
  }

  ASIO_CHECK(bytes_written == sizeof(write_data));
  ASIO_CHECK(bytes_read == sizeof(read_data));
  ASIO_CHECK(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  server_socket.close();
  client_socket.async_read_some(asio::buffer(read_buf), handle_read_eof);
}

ASIO_TEST_SUITE
(
  "buffered_read_stream",
  ASIO_TEST_CASE(test_sync_operations)
  ASIO_TEST_CASE(test_async_operations)
)
