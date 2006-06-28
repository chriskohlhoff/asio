//
// socket_base_test.cpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/socket_base.hpp"

#include "asio.hpp"
#include "unit_test.hpp"

//------------------------------------------------------------------------------

// socket_base_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all nested classes, enums and constants in
// socket_base compile and link correctly. Runtime failures are ignored.

namespace socket_base_compile {

using namespace asio;

void test()
{
  try
  {
    io_service ios;
    ip::tcp::socket sock(ios);
    char buf[1024];

    // shutdown_type enumeration.

    sock.shutdown(socket_base::shutdown_receive);
    sock.shutdown(socket_base::shutdown_send);
    sock.shutdown(socket_base::shutdown_both);

    // message_flags constants.

    sock.receive(buffer(buf), socket_base::message_peek);
    sock.receive(buffer(buf), socket_base::message_out_of_band);
    sock.send(buffer(buf), socket_base::message_do_not_route);

    // broadcast class.

    socket_base::broadcast broadcast1(true);
    sock.set_option(broadcast1);
    socket_base::broadcast broadcast2;
    sock.get_option(broadcast2);

    // do_not_route class.

    socket_base::do_not_route do_not_route1(true);
    sock.set_option(do_not_route1);
    socket_base::do_not_route do_not_route2;
    sock.get_option(do_not_route2);

    // keep_alive class.

    socket_base::keep_alive keep_alive1(true);
    sock.set_option(keep_alive1);
    socket_base::keep_alive keep_alive2;
    sock.get_option(keep_alive2);

    // send_buffer_size class.

    socket_base::send_buffer_size send_buffer_size1(1024);
    sock.set_option(send_buffer_size1);
    socket_base::send_buffer_size send_buffer_size2;
    sock.get_option(send_buffer_size2);

    // send_low_watermark class.

    socket_base::send_low_watermark send_low_watermark1(128);
    sock.set_option(send_low_watermark1);
    socket_base::send_low_watermark send_low_watermark2;
    sock.get_option(send_low_watermark2);

    // receive_buffer_size class.

    socket_base::receive_buffer_size receive_buffer_size1(1024);
    sock.set_option(receive_buffer_size1);
    socket_base::receive_buffer_size receive_buffer_size2;
    sock.get_option(receive_buffer_size2);

    // receive_low_watermark class.

    socket_base::receive_low_watermark receive_low_watermark1(128);
    sock.set_option(receive_low_watermark1);
    socket_base::receive_low_watermark receive_low_watermark2;
    sock.get_option(receive_low_watermark2);

    // reuse_address class.

    socket_base::reuse_address reuse_address1(true);
    sock.set_option(reuse_address1);
    socket_base::reuse_address reuse_address2;
    sock.get_option(reuse_address2);

    // linger class.

    socket_base::linger linger1(true, 30);
    sock.set_option(linger1);
    socket_base::linger linger2;
    sock.get_option(linger2);

    // enable_connection_aborted class.

    socket_base::enable_connection_aborted enable_connection_aborted1(true);
    sock.set_option(enable_connection_aborted1);
    socket_base::enable_connection_aborted enable_connection_aborted2;
    sock.get_option(enable_connection_aborted2);

    // non_blocking_io class.

    socket_base::non_blocking_io non_blocking_io(true);
    sock.io_control(non_blocking_io);

    // bytes_readable class.

    socket_base::bytes_readable bytes_readable;
    sock.io_control(bytes_readable);
    std::size_t bytes = bytes_readable.get();
    (void)bytes;
  }
  catch (std::exception&)
  {
  }
}

} // namespace socket_base_compile

test_suite* init_unit_test_suite(int argc, char* argv[])
{
  test_suite* test = BOOST_TEST_SUITE("socket_base");
  test->add(BOOST_TEST_CASE(&socket_base_compile::test));
  return test;
}
