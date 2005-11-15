//
// buffered_write_stream_test.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Test that header file is self-contained.
#include "asio/buffered_write_stream.hpp"

#include <boost/bind.hpp>
#include <cstring>
#include "asio.hpp"
#include "unit_test.hpp"

typedef asio::buffered_write_stream<
    asio::stream_socket> stream_type;

void test_sync_operations()
{
  using namespace std; // For memcmp.

  asio::demuxer demuxer;

  asio::socket_acceptor acceptor(demuxer,
      asio::ipv4::tcp::endpoint(0));
  asio::ipv4::tcp::endpoint server_endpoint;
  acceptor.get_local_endpoint(server_endpoint);
  server_endpoint.address(asio::ipv4::address::loopback());

  stream_type client_socket(demuxer);
  client_socket.lowest_layer().connect(server_endpoint);

  stream_type server_socket(demuxer);
  acceptor.accept(server_socket);

  const char write_data[]
    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const asio::const_buffer write_buf = asio::buffer(write_data);

  std::size_t bytes_written = 0;
  while (bytes_written < sizeof(write_data))
  {
    bytes_written += client_socket.write_some(
        asio::buffer(write_buf + bytes_written));
    client_socket.flush();
  }

  char read_data[sizeof(write_data)];
  const asio::mutable_buffer read_buf = asio::buffer(read_data);

  std::size_t bytes_read = 0;
  while (bytes_read < sizeof(read_data))
  {
    bytes_read += server_socket.read_some(
        asio::buffer(read_buf + bytes_read));
  }

  BOOST_CHECK(bytes_written == sizeof(write_data));
  BOOST_CHECK(bytes_read == sizeof(read_data));
  BOOST_CHECK(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  bytes_written = 0;
  while (bytes_written < sizeof(write_data))
  {
    bytes_written += server_socket.write_some(
        asio::buffer(write_buf + bytes_written),
        asio::throw_error());
    server_socket.flush(asio::throw_error());
  }

  bytes_read = 0;
  while (bytes_read < sizeof(read_data))
  {
    bytes_read += client_socket.read_some(
        asio::buffer(read_buf + bytes_read),
        asio::throw_error());
  }

  BOOST_CHECK(bytes_written == sizeof(write_data));
  BOOST_CHECK(bytes_read == sizeof(read_data));
  BOOST_CHECK(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  server_socket.close();
  asio::error error;
  bytes_read = client_socket.read_some(
      asio::buffer(read_buf),
      asio::assign_error(error));

  BOOST_CHECK(bytes_read == 0);
  BOOST_CHECK(error == asio::error::eof);

  client_socket.close(asio::throw_error());
}

void handle_accept(const asio::error& e)
{
  BOOST_CHECK(!e);
}

void handle_write(const asio::error& e,
    std::size_t bytes_transferred,
    std::size_t* total_bytes_written)
{
  BOOST_CHECK(!e);
  if (e)
    throw e; // Terminate test.
  *total_bytes_written += bytes_transferred;
}

void handle_flush(const asio::error& e)
{
  BOOST_CHECK(!e);
}

void handle_read(const asio::error& e,
    std::size_t bytes_transferred,
    std::size_t* total_bytes_read)
{
  BOOST_CHECK(!e);
  if (e)
    throw e; // Terminate test.
  *total_bytes_read += bytes_transferred;
}

void handle_read_eof(const asio::error& e,
    std::size_t bytes_transferred)
{
  BOOST_CHECK(e == asio::error::eof);
  BOOST_CHECK(bytes_transferred == 0);
}

void test_async_operations()
{
  using namespace std; // For memcmp.

  asio::demuxer demuxer;

  asio::socket_acceptor acceptor(demuxer,
      asio::ipv4::tcp::endpoint(0));
  asio::ipv4::tcp::endpoint server_endpoint;
  acceptor.get_local_endpoint(server_endpoint);
  server_endpoint.address(asio::ipv4::address::loopback());

  stream_type client_socket(demuxer);
  client_socket.lowest_layer().connect(server_endpoint);

  stream_type server_socket(demuxer);
  acceptor.async_accept(server_socket, handle_accept);
  demuxer.run();
  demuxer.reset();

  const char write_data[]
    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  const asio::const_buffer write_buf = asio::buffer(write_data);

  std::size_t bytes_written = 0;
  while (bytes_written < sizeof(write_data))
  {
    client_socket.async_write_some(
        asio::buffer(write_buf + bytes_written),
        boost::bind(handle_write, asio::placeholders::error,
          asio::placeholders::bytes_transferred, &bytes_written));
    demuxer.run();
    demuxer.reset();
    client_socket.async_flush(
        boost::bind(handle_flush, asio::placeholders::error));
    demuxer.run();
    demuxer.reset();
  }

  char read_data[sizeof(write_data)];
  const asio::mutable_buffer read_buf = asio::buffer(read_data);

  std::size_t bytes_read = 0;
  while (bytes_read < sizeof(read_data))
  {
    server_socket.async_read_some(
        asio::buffer(read_buf + bytes_read),
        boost::bind(handle_read, asio::placeholders::error,
          asio::placeholders::bytes_transferred, &bytes_read));
    demuxer.run();
    demuxer.reset();
  }

  BOOST_CHECK(bytes_written == sizeof(write_data));
  BOOST_CHECK(bytes_read == sizeof(read_data));
  BOOST_CHECK(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  bytes_written = 0;
  while (bytes_written < sizeof(write_data))
  {
    server_socket.async_write_some(
        asio::buffer(write_buf + bytes_written),
        boost::bind(handle_write, asio::placeholders::error,
          asio::placeholders::bytes_transferred, &bytes_written));
    demuxer.run();
    demuxer.reset();
    server_socket.async_flush(
        boost::bind(handle_flush, asio::placeholders::error));
    demuxer.run();
    demuxer.reset();
  }

  bytes_read = 0;
  while (bytes_read < sizeof(read_data))
  {
    client_socket.async_read_some(
        asio::buffer(read_buf + bytes_read),
        boost::bind(handle_read, asio::placeholders::error,
          asio::placeholders::bytes_transferred, &bytes_read));
    demuxer.run();
    demuxer.reset();
  }

  BOOST_CHECK(bytes_written == sizeof(write_data));
  BOOST_CHECK(bytes_read == sizeof(read_data));
  BOOST_CHECK(memcmp(write_data, read_data, sizeof(write_data)) == 0);

  server_socket.close();
  client_socket.async_read_some(asio::buffer(read_buf), handle_read_eof);
}

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("buffered_write_stream");
  test->add(BOOST_TEST_CASE(&test_sync_operations));
  test->add(BOOST_TEST_CASE(&test_async_operations));
  return test;
}
