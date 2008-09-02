//
// stream_descriptor.cpp
// ~~~~~~~~~~~~~~~~~~~~~
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
#include "asio/posix/stream_descriptor.hpp"

#include "asio.hpp"
#include "../unit_test.hpp"

//------------------------------------------------------------------------------

// posix_stream_descriptor_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// posix::stream_descriptor compile and link correctly. Runtime failures are
// ignored.

namespace posix_stream_descriptor_compile {

void write_some_handler(const asio::error_code&, std::size_t)
{
}

void read_some_handler(const asio::error_code&, std::size_t)
{
}

void test()
{
#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR)
  using namespace asio;
  namespace posix = asio::posix;

  try
  {
    io_service ios;
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    posix::descriptor_base::bytes_readable io_control_command;
    asio::error_code ec;

    // basic_stream_descriptor constructors.

    posix::stream_descriptor descriptor1(ios);
    int native_descriptor1 = -1;
    posix::stream_descriptor descriptor2(ios, native_descriptor1);

    // basic_io_object functions.

    io_service& ios_ref = descriptor1.io_service();
    (void)ios_ref;

    // basic_descriptor functions.

    posix::stream_descriptor::lowest_layer_type& lowest_layer
      = descriptor1.lowest_layer();
    (void)lowest_layer;

    const posix::stream_descriptor& descriptor3 = descriptor1;
    const posix::stream_descriptor::lowest_layer_type& lowest_layer2
      = descriptor3.lowest_layer();
    (void)lowest_layer2;

    int native_descriptor2 = -1;
    descriptor1.assign(native_descriptor2);

    bool is_open = descriptor1.is_open();
    (void)is_open;

    descriptor1.close();
    descriptor1.close(ec);

    posix::stream_descriptor::native_type native_descriptor3
      = descriptor1.native();
    (void)native_descriptor3;

    descriptor1.cancel();
    descriptor1.cancel(ec);

    descriptor1.io_control(io_control_command);
    descriptor1.io_control(io_control_command, ec);

    // basic_stream_descriptor functions.

    descriptor1.write_some(buffer(mutable_char_buffer));
    descriptor1.write_some(buffer(const_char_buffer));
    descriptor1.write_some(null_buffers());
    descriptor1.write_some(buffer(mutable_char_buffer), ec);
    descriptor1.write_some(buffer(const_char_buffer), ec);
    descriptor1.write_some(null_buffers(), ec);

    descriptor1.async_write_some(buffer(mutable_char_buffer),
        write_some_handler);
    descriptor1.async_write_some(buffer(const_char_buffer),
        write_some_handler);
    descriptor1.async_write_some(null_buffers(),
        write_some_handler);

    descriptor1.read_some(buffer(mutable_char_buffer));
    descriptor1.read_some(buffer(mutable_char_buffer), ec);
    descriptor1.read_some(null_buffers(), ec);

    descriptor1.async_read_some(buffer(mutable_char_buffer), read_some_handler);
    descriptor1.async_read_some(null_buffers(), read_some_handler);
  }
  catch (std::exception&)
  {
  }
#endif // defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR)
}

} // namespace posix_stream_descriptor_compile

//------------------------------------------------------------------------------
test_suite* init_unit_test_suite(int, char*[])
{
  test_suite* test = BOOST_TEST_SUITE("posix/stream_descriptor");
  test->add(BOOST_TEST_CASE(&posix_stream_descriptor_compile::test));
  return test;
}
