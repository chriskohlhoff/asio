//
// udp.cpp
// ~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

// Disable autolinking for unit tests.
#if !defined(BOOST_ALL_NO_LIB)
#define BOOST_ALL_NO_LIB 1
#endif // !defined(BOOST_ALL_NO_LIB)

// Test that header file is self-contained.
#include "asio/ip/udp.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <thread>
#if defined(ASIO_HAS_STD_RANGES)
# include <ranges>
#endif // defined(ASIO_HAS_STD_RANGES)
#include "asio/deferred.hpp"
#include "asio/io_context.hpp"
#include "../unit_test.hpp"
#include "../archetypes/async_result.hpp"
#include "../archetypes/gettable_socket_option.hpp"
#include "../archetypes/io_control_command.hpp"
#include "../archetypes/settable_socket_option.hpp"

//------------------------------------------------------------------------------

// ip_udp_socket_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ip::udp::socket compile and link correctly. Runtime failures are ignored.

namespace ip_udp_socket_compile {

struct connect_handler
{
  connect_handler() {}
  void operator()(const asio::error_code&) {}
  connect_handler(connect_handler&&) {}
private:
  connect_handler(const connect_handler&);
};

struct wait_handler
{
  wait_handler() {}
  void operator()(const asio::error_code&) {}
  wait_handler(wait_handler&&) {}
private:
  wait_handler(const wait_handler&);
};

struct send_handler
{
  send_handler() {}
  void operator()(const asio::error_code&, std::size_t) {}
  send_handler(send_handler&&) {}
private:
  send_handler(const send_handler&);
};

struct receive_handler
{
  receive_handler() {}
  void operator()(const asio::error_code&, std::size_t) {}
  receive_handler(receive_handler&&) {}
private:
  receive_handler(const receive_handler&);
};

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  try
  {
    io_context ioc;
    const io_context::executor_type ioc_ex = ioc.get_executor();
    char mutable_char_buffer[128] = "";
    const char const_char_buffer[128] = "";
    socket_base::message_flags in_flags = 0;
    archetypes::settable_socket_option<void> settable_socket_option1;
    archetypes::settable_socket_option<int> settable_socket_option2;
    archetypes::settable_socket_option<double> settable_socket_option3;
    archetypes::gettable_socket_option<void> gettable_socket_option1;
    archetypes::gettable_socket_option<int> gettable_socket_option2;
    archetypes::gettable_socket_option<double> gettable_socket_option3;
    archetypes::io_control_command io_control_command;
    archetypes::immediate_handler immediate;
    archetypes::lazy_handler lazy;
    asio::error_code ec;

    // basic_datagram_socket constructors.

    ip::udp::socket socket1(ioc);
    ip::udp::socket socket2(ioc, ip::udp::v4());
    ip::udp::socket socket3(ioc, ip::udp::v6());
    ip::udp::socket socket4(ioc, ip::udp::endpoint(ip::udp::v4(), 0));
    ip::udp::socket socket5(ioc, ip::udp::endpoint(ip::udp::v6(), 0));
#if !defined(ASIO_WINDOWS_RUNTIME)
    ip::udp::socket::native_handle_type native_socket1
      = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    ip::udp::socket socket6(ioc, ip::udp::v4(), native_socket1);
#endif // !defined(ASIO_WINDOWS_RUNTIME)

    ip::udp::socket socket7(ioc_ex);
    ip::udp::socket socket8(ioc_ex, ip::udp::v4());
    ip::udp::socket socket9(ioc_ex, ip::udp::v6());
    ip::udp::socket socket10(ioc_ex, ip::udp::endpoint(ip::udp::v4(), 0));
    ip::udp::socket socket11(ioc_ex, ip::udp::endpoint(ip::udp::v6(), 0));
#if !defined(ASIO_WINDOWS_RUNTIME)
    ip::udp::socket::native_handle_type native_socket2
      = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    ip::udp::socket socket12(ioc_ex, ip::udp::v4(), native_socket2);
#endif // !defined(ASIO_WINDOWS_RUNTIME)

    ip::udp::socket socket13(std::move(socket6));

    // basic_datagram_socket operators.

    socket1 = ip::udp::socket(ioc);
    socket1 = std::move(socket2);

    // I/O object functions.

    ip::udp::socket::executor_type ex = socket1.get_executor();
    (void)ex;

    // basic_socket functions.

    ip::udp::socket::lowest_layer_type& lowest_layer = socket1.lowest_layer();
    (void)lowest_layer;

    const ip::udp::socket& socket14 = socket1;
    const ip::udp::socket::lowest_layer_type& lowest_layer2
      = socket14.lowest_layer();
    (void)lowest_layer2;

    socket1.open(ip::udp::v4());
    socket1.open(ip::udp::v6());
    socket1.open(ip::udp::v4(), ec);
    socket1.open(ip::udp::v6(), ec);

#if !defined(ASIO_WINDOWS_RUNTIME)
    ip::udp::socket::native_handle_type native_socket3
      = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    socket1.assign(ip::udp::v4(), native_socket3);
    ip::udp::socket::native_handle_type native_socket4
      = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    socket1.assign(ip::udp::v4(), native_socket4, ec);
#endif // !defined(ASIO_WINDOWS_RUNTIME)

    bool is_open = socket1.is_open();
    (void)is_open;

    socket1.close();
    socket1.close(ec);

    socket1.release();
    socket1.release(ec);

    ip::udp::socket::native_handle_type native_socket5
      = socket1.native_handle();
    (void)native_socket5;

    socket1.cancel();
    socket1.cancel(ec);

    bool at_mark1 = socket1.at_mark();
    (void)at_mark1;
    bool at_mark2 = socket1.at_mark(ec);
    (void)at_mark2;

    std::size_t available1 = socket1.available();
    (void)available1;
    std::size_t available2 = socket1.available(ec);
    (void)available2;

    socket1.bind(ip::udp::endpoint(ip::udp::v4(), 0));
    socket1.bind(ip::udp::endpoint(ip::udp::v6(), 0));
    socket1.bind(ip::udp::endpoint(ip::udp::v4(), 0), ec);
    socket1.bind(ip::udp::endpoint(ip::udp::v6(), 0), ec);

    socket1.connect(ip::udp::endpoint(ip::udp::v4(), 0));
    socket1.connect(ip::udp::endpoint(ip::udp::v6(), 0));
    socket1.connect(ip::udp::endpoint(ip::udp::v4(), 0), ec);
    socket1.connect(ip::udp::endpoint(ip::udp::v6(), 0), ec);

    socket1.async_connect(ip::udp::endpoint(ip::udp::v4(), 0),
        connect_handler());
    socket1.async_connect(ip::udp::endpoint(ip::udp::v6(), 0),
        connect_handler());
    socket1.async_connect(ip::udp::endpoint(ip::udp::v4(), 0), immediate);
    socket1.async_connect(ip::udp::endpoint(ip::udp::v6(), 0), immediate);
    int i1 = socket1.async_connect(ip::udp::endpoint(ip::udp::v4(), 0), lazy);
    (void)i1;
    int i2 = socket1.async_connect(ip::udp::endpoint(ip::udp::v6(), 0), lazy);
    (void)i2;

    socket1.set_option(settable_socket_option1);
    socket1.set_option(settable_socket_option1, ec);
    socket1.set_option(settable_socket_option2);
    socket1.set_option(settable_socket_option2, ec);
    socket1.set_option(settable_socket_option3);
    socket1.set_option(settable_socket_option3, ec);

    socket1.get_option(gettable_socket_option1);
    socket1.get_option(gettable_socket_option1, ec);
    socket1.get_option(gettable_socket_option2);
    socket1.get_option(gettable_socket_option2, ec);
    socket1.get_option(gettable_socket_option3);
    socket1.get_option(gettable_socket_option3, ec);

    socket1.io_control(io_control_command);
    socket1.io_control(io_control_command, ec);

    bool non_blocking1 = socket1.non_blocking();
    (void)non_blocking1;
    socket1.non_blocking(true);
    socket1.non_blocking(false, ec);

    bool non_blocking2 = socket1.native_non_blocking();
    (void)non_blocking2;
    socket1.native_non_blocking(true);
    socket1.native_non_blocking(false, ec);

    ip::udp::endpoint endpoint1 = socket1.local_endpoint();
    (void)endpoint1;
    ip::udp::endpoint endpoint2 = socket1.local_endpoint(ec);
    (void)endpoint2;

    ip::udp::endpoint endpoint3 = socket1.remote_endpoint();
    (void)endpoint3;
    ip::udp::endpoint endpoint4 = socket1.remote_endpoint(ec);
    (void)endpoint4;

    socket1.shutdown(socket_base::shutdown_both);
    socket1.shutdown(socket_base::shutdown_both, ec);

    socket1.wait(socket_base::wait_read);
    socket1.wait(socket_base::wait_write, ec);

    socket1.async_wait(socket_base::wait_read, wait_handler());
    socket1.async_wait(socket_base::wait_read, immediate);
    int i3 = socket1.async_wait(socket_base::wait_write, lazy);
    (void)i3;

    // basic_datagram_socket functions.

    socket1.send(buffer(mutable_char_buffer));
    socket1.send(buffer(const_char_buffer));
    socket1.send(null_buffers());
    socket1.send(buffer(mutable_char_buffer), in_flags);
    socket1.send(buffer(const_char_buffer), in_flags);
    socket1.send(null_buffers(), in_flags);
    socket1.send(buffer(mutable_char_buffer), in_flags, ec);
    socket1.send(buffer(const_char_buffer), in_flags, ec);
    socket1.send(null_buffers(), in_flags, ec);

    socket1.async_send(buffer(mutable_char_buffer), send_handler());
    socket1.async_send(buffer(const_char_buffer), send_handler());
    socket1.async_send(null_buffers(), send_handler());
    socket1.async_send(buffer(mutable_char_buffer), in_flags, send_handler());
    socket1.async_send(buffer(const_char_buffer), in_flags, send_handler());
    socket1.async_send(null_buffers(), in_flags, send_handler());
    socket1.async_send(buffer(mutable_char_buffer), immediate);
    socket1.async_send(buffer(const_char_buffer), immediate);
    socket1.async_send(null_buffers(), immediate);
    socket1.async_send(buffer(mutable_char_buffer), in_flags, immediate);
    socket1.async_send(buffer(const_char_buffer), in_flags, immediate);
    socket1.async_send(null_buffers(), in_flags, immediate);
    int i4 = socket1.async_send(buffer(mutable_char_buffer), lazy);
    (void)i4;
    int i5 = socket1.async_send(buffer(const_char_buffer), lazy);
    (void)i5;
    int i6 = socket1.async_send(null_buffers(), lazy);
    (void)i6;
    int i7 = socket1.async_send(buffer(mutable_char_buffer), in_flags, lazy);
    (void)i7;
    int i8 = socket1.async_send(buffer(const_char_buffer), in_flags, lazy);
    (void)i8;
    int i9 = socket1.async_send(null_buffers(), in_flags, lazy);
    (void)i9;

    socket1.send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0));
    socket1.send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0));
    socket1.send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0));
    socket1.send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0));
    socket1.send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v4(), 0));
    socket1.send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v6(), 0));
    socket1.send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags);
    socket1.send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags);
    socket1.send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags);
    socket1.send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags);
    socket1.send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags);
    socket1.send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags);
    socket1.send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, ec);
    socket1.send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, ec);
    socket1.send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, ec);
    socket1.send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, ec);
    socket1.send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, ec);
    socket1.send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, ec);

    socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), send_handler());
    socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), send_handler());
    socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), send_handler());
    socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), send_handler());
    socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v4(), 0), send_handler());
    socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v6(), 0), send_handler());
    socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, send_handler());
    socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, send_handler());
    socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, send_handler());
    socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, send_handler());
    socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, send_handler());
    socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, send_handler());
    socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), immediate);
    socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), immediate);
    socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), immediate);
    socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), immediate);
    socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v4(), 0), immediate);
    socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v6(), 0), immediate);
    socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, immediate);
    socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, immediate);
    socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, immediate);
    socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, immediate);
    socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, immediate);
    socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, immediate);
    int i10 = socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), lazy);
    (void)i10;
    int i11 = socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), lazy);
    (void)i11;
    int i12 = socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), lazy);
    (void)i12;
    int i13 = socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), lazy);
    (void)i13;
    int i14 = socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v4(), 0), lazy);
    (void)i14;
    int i15 = socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v6(), 0), lazy);
    (void)i15;
    int i16 = socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, lazy);
    (void)i16;
    int i17 = socket1.async_send_to(buffer(mutable_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, lazy);
    (void)i17;
    int i18 = socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, lazy);
    (void)i18;
    int i19 = socket1.async_send_to(buffer(const_char_buffer),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, lazy);
    (void)i19;
    int i20 = socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v4(), 0), in_flags, lazy);
    (void)i20;
    int i21 = socket1.async_send_to(null_buffers(),
        ip::udp::endpoint(ip::udp::v6(), 0), in_flags, lazy);
    (void)i21;

    socket1.receive(buffer(mutable_char_buffer));
    socket1.receive(null_buffers());
    socket1.receive(buffer(mutable_char_buffer), in_flags);
    socket1.receive(null_buffers(), in_flags);
    socket1.receive(buffer(mutable_char_buffer), in_flags, ec);
    socket1.receive(null_buffers(), in_flags, ec);

    socket1.async_receive(buffer(mutable_char_buffer), receive_handler());
    socket1.async_receive(null_buffers(), receive_handler());
    socket1.async_receive(buffer(mutable_char_buffer), in_flags,
        receive_handler());
    socket1.async_receive(null_buffers(), in_flags, receive_handler());
    socket1.async_receive(buffer(mutable_char_buffer), immediate);
    socket1.async_receive(null_buffers(), immediate);
    socket1.async_receive(buffer(mutable_char_buffer), in_flags, immediate);
    socket1.async_receive(null_buffers(), in_flags, immediate);
    int i22 = socket1.async_receive(buffer(mutable_char_buffer), lazy);
    (void)i22;
    int i23 = socket1.async_receive(null_buffers(), lazy);
    (void)i23;
    int i24 = socket1.async_receive(buffer(mutable_char_buffer),
        in_flags, lazy);
    (void)i24;
    int i25 = socket1.async_receive(null_buffers(), in_flags, lazy);
    (void)i25;

    ip::udp::endpoint endpoint;
    socket1.receive_from(buffer(mutable_char_buffer), endpoint);
    socket1.receive_from(null_buffers(), endpoint);
    socket1.receive_from(buffer(mutable_char_buffer), endpoint, in_flags);
    socket1.receive_from(null_buffers(), endpoint, in_flags);
    socket1.receive_from(buffer(mutable_char_buffer), endpoint, in_flags, ec);
    socket1.receive_from(null_buffers(), endpoint, in_flags, ec);

    socket1.async_receive_from(buffer(mutable_char_buffer),
        endpoint, receive_handler());
    socket1.async_receive_from(null_buffers(),
        endpoint, receive_handler());
    socket1.async_receive_from(buffer(mutable_char_buffer),
        endpoint, in_flags, receive_handler());
    socket1.async_receive_from(null_buffers(),
        endpoint, in_flags, receive_handler());
    socket1.async_receive_from(buffer(mutable_char_buffer),
        endpoint, immediate);
    socket1.async_receive_from(null_buffers(),
        endpoint, immediate);
    socket1.async_receive_from(buffer(mutable_char_buffer),
        endpoint, in_flags, immediate);
    socket1.async_receive_from(null_buffers(),
        endpoint, in_flags, immediate);
    int i26 = socket1.async_receive_from(buffer(mutable_char_buffer),
        endpoint, lazy);
    (void)i26;
    int i27 = socket1.async_receive_from(null_buffers(),
        endpoint, lazy);
    (void)i27;
    int i28 = socket1.async_receive_from(buffer(mutable_char_buffer),
        endpoint, in_flags, lazy);
    (void)i28;
    int i29 = socket1.async_receive_from(null_buffers(),
        endpoint, in_flags, lazy);
    (void)i29;

    std::array<asio::const_buffer, 1> send_bufs = {
        { buffer(const_char_buffer) } };
    std::array<asio::mutable_buffer, 1> recv_bufs = {
        { buffer(mutable_char_buffer) } };
    std::array<ip::udp::endpoint, 1> multi_endpoints = {
        { ip::udp::endpoint(ip::udp::v4(), 0) } };
    std::array<std::size_t, 1> multi_sizes = { { 0 } };
    std::array<socket_base::message_flags, 1> multi_flags = { { 0 } };
    const std::array<ip::udp::endpoint, 2> const_larger_endpoints = {
        { ip::udp::endpoint(ip::udp::v4(), 0),
          ip::udp::endpoint(ip::udp::v4(), 0) } };
    std::array<ip::udp::endpoint, 2> larger_endpoints = {
        { ip::udp::endpoint(ip::udp::v4(), 0),
          ip::udp::endpoint(ip::udp::v4(), 0) } };
    std::array<std::size_t, 2> larger_sizes = { { 0, 0 } };
    std::size_t ref_size = 0;
    ip::udp::endpoint ref_endpoint(ip::udp::v4(), 0);
    std::array<std::reference_wrapper<std::size_t>, 1> ref_sizes = {
        { std::ref(ref_size) } };
    std::array<std::reference_wrapper<ip::udp::endpoint>, 1> ref_endpoints = {
        { std::ref(ref_endpoint) } };

    socket1.send_multiple(send_bufs);
    socket1.send_multiple(send_bufs, in_flags);
    socket1.send_multiple(send_bufs, in_flags, ec);

    socket1.async_send_multiple(send_bufs, send_handler());
    socket1.async_send_multiple(send_bufs, in_flags, send_handler());
    socket1.async_send_multiple(send_bufs, immediate);
    socket1.async_send_multiple(send_bufs, in_flags, immediate);
    int i30 = socket1.async_send_multiple(send_bufs, lazy);
    (void)i30;
    int i31 = socket1.async_send_multiple(send_bufs, in_flags, lazy);
    (void)i31;

    socket1.send_to_multiple(send_bufs, multi_endpoints);
    socket1.send_to_multiple(send_bufs, multi_endpoints, in_flags);
    socket1.send_to_multiple(send_bufs, multi_endpoints, in_flags, ec);

    socket1.async_send_to_multiple(send_bufs, multi_endpoints, send_handler());
    socket1.async_send_to_multiple(send_bufs, multi_endpoints, in_flags,
        send_handler());
    socket1.async_send_to_multiple(send_bufs, multi_endpoints, immediate);
    socket1.async_send_to_multiple(send_bufs, multi_endpoints, in_flags,
        immediate);
    int i32 = socket1.async_send_to_multiple(send_bufs, multi_endpoints, lazy);
    (void)i32;
    int i33 = socket1.async_send_to_multiple(send_bufs, multi_endpoints,
        in_flags, lazy);
    (void)i33;

    socket1.send_to_multiple(send_bufs, const_larger_endpoints);
    socket1.send_to_multiple(send_bufs, const_larger_endpoints, in_flags, ec);
    socket1.async_send_to_multiple(send_bufs, const_larger_endpoints,
        send_handler());
    socket1.async_send_to_multiple(send_bufs, const_larger_endpoints,
        in_flags, immediate);

    socket1.receive_multiple(recv_bufs, std::ref(multi_sizes));
    socket1.receive_multiple(recv_bufs, std::ref(multi_sizes), in_flags);
    socket1.receive_multiple(recv_bufs, std::ref(multi_sizes), in_flags, ec);
    socket1.receive_multiple(recv_bufs, std::ref(multi_sizes),
        std::ref(multi_flags));
    socket1.receive_multiple(recv_bufs, std::ref(multi_sizes), in_flags,
        std::ref(multi_flags));
    socket1.receive_multiple(recv_bufs, std::ref(multi_sizes), in_flags,
        std::ref(multi_flags), ec);

    socket1.async_receive_multiple(recv_bufs, std::ref(multi_sizes),
        receive_handler());
    socket1.async_receive_multiple(recv_bufs, std::ref(multi_sizes), in_flags,
        receive_handler());
    socket1.async_receive_multiple(recv_bufs, std::ref(multi_sizes),
        std::ref(multi_flags), receive_handler());
    socket1.async_receive_multiple(recv_bufs, std::ref(multi_sizes), in_flags,
        std::ref(multi_flags), receive_handler());
    socket1.async_receive_multiple(recv_bufs, std::ref(multi_sizes),
        immediate);
    socket1.async_receive_multiple(recv_bufs, std::ref(multi_sizes), in_flags,
        immediate);
    socket1.async_receive_multiple(recv_bufs, std::ref(multi_sizes),
        std::ref(multi_flags), immediate);
    socket1.async_receive_multiple(recv_bufs, std::ref(multi_sizes), in_flags,
        std::ref(multi_flags), immediate);
    int i34 = socket1.async_receive_multiple(recv_bufs,
        std::ref(multi_sizes), lazy);
    (void)i34;
    int i35 = socket1.async_receive_multiple(recv_bufs,
        std::ref(multi_sizes), in_flags, lazy);
    (void)i35;
    int i36 = socket1.async_receive_multiple(recv_bufs,
        std::ref(multi_sizes), std::ref(multi_flags), lazy);
    (void)i36;
    int i37 = socket1.async_receive_multiple(recv_bufs,
        std::ref(multi_sizes), in_flags, std::ref(multi_flags), lazy);
    (void)i37;

    socket1.receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes));
    socket1.receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), in_flags);
    socket1.receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), in_flags, ec);
    socket1.receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), std::ref(multi_flags));
    socket1.receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), in_flags, std::ref(multi_flags));
    socket1.receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), in_flags, std::ref(multi_flags), ec);

    socket1.async_receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), receive_handler());
    socket1.async_receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), in_flags, receive_handler());
    socket1.async_receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), std::ref(multi_flags), receive_handler());
    socket1.async_receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), in_flags, std::ref(multi_flags),
        receive_handler());
    socket1.async_receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), immediate);
    socket1.async_receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), in_flags, immediate);
    socket1.async_receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), std::ref(multi_flags), immediate);
    socket1.async_receive_from_multiple(recv_bufs, std::ref(multi_endpoints),
        std::ref(multi_sizes), in_flags, std::ref(multi_flags), immediate);
    int i38 = socket1.async_receive_from_multiple(recv_bufs,
        std::ref(multi_endpoints), std::ref(multi_sizes), lazy);
    (void)i38;
    int i39 = socket1.async_receive_from_multiple(recv_bufs,
        std::ref(multi_endpoints), std::ref(multi_sizes), in_flags, lazy);
    (void)i39;
    int i40 = socket1.async_receive_from_multiple(recv_bufs,
        std::ref(multi_endpoints), std::ref(multi_sizes),
        std::ref(multi_flags), lazy);
    (void)i40;
    int i41 = socket1.async_receive_from_multiple(recv_bufs,
        std::ref(multi_endpoints), std::ref(multi_sizes), in_flags,
        std::ref(multi_flags), lazy);
    (void)i41;

    socket1.receive_multiple(recv_bufs, std::ref(larger_sizes));
    socket1.async_receive_multiple(recv_bufs, std::ref(larger_sizes),
        receive_handler());
    socket1.receive_from_multiple(recv_bufs, std::ref(larger_endpoints),
        std::ref(larger_sizes));
    socket1.async_receive_from_multiple(recv_bufs, std::ref(larger_endpoints),
        std::ref(larger_sizes), receive_handler());

    socket1.send_to_multiple(std::cref(send_bufs), std::cref(multi_endpoints));
    socket1.async_send_to_multiple(std::cref(send_bufs),
        std::cref(multi_endpoints), send_handler());
    socket1.receive_from_multiple(std::ref(recv_bufs),
        std::ref(multi_endpoints), std::ref(multi_sizes));
    socket1.async_receive_from_multiple(std::ref(recv_bufs),
        std::ref(multi_endpoints), std::ref(multi_sizes), receive_handler());

    socket1.send_to_multiple(send_bufs, ref_endpoints);
    socket1.async_send_to_multiple(send_bufs, ref_endpoints, send_handler());
    socket1.receive_from_multiple(recv_bufs, ref_endpoints, ref_sizes);
    socket1.async_receive_from_multiple(recv_bufs, ref_endpoints, ref_sizes,
        receive_handler());
  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_udp_socket_compile

//------------------------------------------------------------------------------

// ip_udp_socket_runtime test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks the runtime operation of the ip::udp::socket class.

namespace ip_udp_socket_runtime {

void handle_send(size_t expected_bytes_sent,
    const asio::error_code& err, size_t bytes_sent)
{
  ASIO_CHECK(!err);
  ASIO_CHECK(expected_bytes_sent == bytes_sent);
}

void handle_recv(size_t expected_bytes_recvd,
    const asio::error_code& err, size_t bytes_recvd)
{
  ASIO_CHECK(!err);
  ASIO_CHECK(expected_bytes_recvd == bytes_recvd);
}

void test()
{
  using namespace std; // For memcmp and memset.
  using namespace asio;
  namespace ip = asio::ip;

  namespace bindns = std;
  using bindns::placeholders::_1;
  using bindns::placeholders::_2;

  io_context ioc;

  ip::udp::socket s1(ioc, ip::udp::endpoint(ip::udp::v4(), 0));
  ip::udp::endpoint target_endpoint = s1.local_endpoint();
  target_endpoint.address(ip::address_v4::loopback());

  ip::udp::socket s2(ioc);
  s2.open(ip::udp::v4());
  s2.bind(ip::udp::endpoint(ip::udp::v4(), 0));
  char send_msg[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  s2.send_to(buffer(send_msg, sizeof(send_msg)), target_endpoint);

  char recv_msg[sizeof(send_msg)];
  ip::udp::endpoint sender_endpoint;
  size_t bytes_recvd = s1.receive_from(buffer(recv_msg, sizeof(recv_msg)),
      sender_endpoint);

  ASIO_CHECK(bytes_recvd == sizeof(send_msg));
  ASIO_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);

  memset(recv_msg, 0, sizeof(recv_msg));

  target_endpoint = sender_endpoint;
  s1.async_send_to(buffer(send_msg, sizeof(send_msg)), target_endpoint,
      bindns::bind(handle_send, sizeof(send_msg), _1, _2));
  s2.async_receive_from(buffer(recv_msg, sizeof(recv_msg)), sender_endpoint,
      bindns::bind(handle_recv, sizeof(recv_msg), _1, _2));

  ioc.run();

  ASIO_CHECK(memcmp(send_msg, recv_msg, sizeof(send_msg)) == 0);
}

} // namespace ip_udp_socket_runtime

//------------------------------------------------------------------------------

// ip_udp_socket_multiple_runtime test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks the runtime operation of the multiple-datagram
// send and receive functions of the ip::udp::socket class.

namespace ip_udp_socket_multiple_runtime {

// A range that refers to elements held elsewhere, similar to a span.
template <typename T>
class test_span
{
public:
  test_span(T* data, std::size_t size)
    : data_(data),
      size_(size)
  {
  }

  T* begin() const
  {
    return data_;
  }

  T* end() const
  {
    return data_ + size_;
  }

private:
  T* data_;
  std::size_t size_;
};

template <typename T>
test_span<T> make_span(T* data, std::size_t size)
{
  return test_span<T>(data, size);
}

// A range that yields copies of a single endpoint.
class test_value_range
{
public:
  class iterator
  {
  public:
    iterator(const asio::ip::udp::endpoint* endpoint, std::size_t index)
      : endpoint_(endpoint),
        index_(index)
    {
    }

    asio::ip::udp::endpoint operator*() const
    {
      return *endpoint_;
    }

    iterator& operator++()
    {
      ++index_;
      return *this;
    }

    bool operator!=(const iterator& other) const
    {
      return index_ != other.index_;
    }

  private:
    const asio::ip::udp::endpoint* endpoint_;
    std::size_t index_;
  };

  test_value_range(const asio::ip::udp::endpoint& endpoint, std::size_t size)
    : endpoint_(endpoint),
      size_(size)
  {
  }

  iterator begin() const
  {
    return iterator(&endpoint_, 0);
  }

  iterator end() const
  {
    return iterator(&endpoint_, size_);
  }

private:
  asio::ip::udp::endpoint endpoint_;
  std::size_t size_;
};

} // namespace ip_udp_socket_multiple_runtime

namespace asio {

template <typename T>
struct is_shallow_mutable_sequence<
    ip_udp_socket_multiple_runtime::test_span<T>>
  : true_type
{
};

} // namespace asio

namespace ip_udp_socket_multiple_runtime {

// Retry a synchronous operation on a non-blocking socket, with a bounded wait.
template <typename Function>
std::size_t retry(Function f, asio::error_code& ec)
{
  for (int i = 0; i < 5000; ++i)
  {
    std::size_t n = f(ec);
    if (ec != asio::error::would_block)
      return n;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return 0;
}

struct async_result
{
  async_result()
    : called(false),
      n(0)
  {
  }

  bool called;
  asio::error_code ec;
  std::size_t n;
};

struct async_handler
{
  explicit async_handler(async_result* r)
    : r_(r)
  {
  }

  void operator()(const asio::error_code& ec, std::size_t n)
  {
    r_->called = true;
    r_->ec = ec;
    r_->n = n;
  }

  async_result* r_;
};

// The number of datagrams a single call may transfer.
#if defined(ASIO_HAS_RECVMMSG_SENDMMSG)
const std::size_t expected_batch = 3;
#else // defined(ASIO_HAS_RECVMMSG_SENDMMSG)
const std::size_t expected_batch = 1;
#endif // defined(ASIO_HAS_RECVMMSG_SENDMMSG)

void test()
{
  using namespace std; // For memcmp.
  using namespace asio;
  namespace ip = asio::ip;

  io_context ioc;
  asio::error_code ec;

  ip::udp::endpoint loopback(ip::address_v4::loopback(), 0);

  ip::udp::socket r(ioc, loopback);
  r.non_blocking(true);
  ip::udp::endpoint r_ep = r.local_endpoint();

  ip::udp::socket a(ioc, loopback);
  ip::udp::socket b(ioc, loopback);

  ip::udp::socket c1(ioc, loopback);
  ip::udp::socket c2(ioc, loopback);
  c1.connect(c2.local_endpoint());
  c2.connect(c1.local_endpoint());
  c1.non_blocking(true);

  char data[100];
  for (std::size_t i = 0; i < sizeof(data); ++i)
    data[i] = static_cast<char>(i);

  char recv_data[8][100];
  std::array<mutable_buffer, 8> recv_bufs;
  for (std::size_t i = 0; i < recv_bufs.size(); ++i)
    recv_bufs[i] = buffer(recv_data[i]);

  std::array<std::size_t, 8> sizes;
  std::array<ip::udp::endpoint, 8> endpoints;
  std::array<socket_base::message_flags, 8> out_flags;

  // Send and receive on a connected socket.

  std::array<const_buffer, 3> send_bufs = { {
      buffer(data, 10), buffer(data, 20), buffer(data, 30) } };
  std::size_t sent = 0;
  std::size_t first = 0;
  while (sent < send_bufs.size())
  {
    std::size_t n = c2.send_multiple(
        make_span(send_bufs.data() + sent, send_bufs.size() - sent), 0, ec);
    ASIO_CHECK(!ec);
    ASIO_CHECK(n > 0);
    if (sent == 0)
      first = n;
    sent += n;
  }
  ASIO_CHECK(first == expected_batch);

  sizes.fill(0);
  std::size_t received = 0;
  while (received < 3)
  {
    std::size_t n = retry(
        [&](asio::error_code& e)
        {
          return c1.receive_multiple(
              make_span(recv_bufs.data() + received, 8 - received),
              make_span(sizes.data() + received, 8 - received), 0, e);
        }, ec);
    ASIO_CHECK(!ec);
    if (ec)
      break;
    received += n;
  }
  ASIO_CHECK(received == 3);
  ASIO_CHECK(sizes[0] == 10);
  ASIO_CHECK(sizes[1] == 20);
  ASIO_CHECK(sizes[2] == 30);
  ASIO_CHECK(memcmp(recv_data[2], data, 30) == 0);

  // Receive from two senders.

  a.send_to(buffer(data, 5), r_ep);
  b.send_to(buffer(data, 6), r_ep);
  a.send_to(buffer(data, 7), r_ep);

  sizes.fill(0);
  received = 0;
  while (received < 3)
  {
    std::size_t n = retry(
        [&](asio::error_code& e)
        {
          return r.receive_from_multiple(
              make_span(recv_bufs.data() + received, 8 - received),
              make_span(endpoints.data() + received, 8 - received),
              make_span(sizes.data() + received, 8 - received), 0, e);
        }, ec);
    ASIO_CHECK(!ec);
    if (ec)
      break;
    received += n;
  }
  ASIO_CHECK(received == 3);
  ASIO_CHECK(sizes[0] == 5);
  ASIO_CHECK(sizes[1] == 6);
  ASIO_CHECK(sizes[2] == 7);
  ASIO_CHECK(endpoints[0].port() == a.local_endpoint().port());
  ASIO_CHECK(endpoints[1].port() == b.local_endpoint().port());
  ASIO_CHECK(endpoints[2].port() == a.local_endpoint().port());
  ASIO_CHECK(endpoints[0].address() == ip::address_v4::loopback());

  // Truncated datagrams.

  std::array<mutable_buffer, 2> small_bufs = { {
      buffer(recv_data[0], 10), buffer(recv_data[1], 10) } };

  a.send_to(buffer(data, 100), r_ep);
  a.send_to(buffer(data, 4), r_ep);

  sizes.fill(0);
  out_flags.fill(0);
  received = 0;
  while (received < 2)
  {
    std::size_t n = retry(
        [&](asio::error_code& e)
        {
          return r.receive_from_multiple(
              make_span(small_bufs.data() + received, 2 - received),
              make_span(endpoints.data() + received, 2 - received),
              make_span(sizes.data() + received, 2 - received), 0,
              make_span(out_flags.data() + received, 2 - received), e);
        }, ec);
    ASIO_CHECK(!ec);
    if (ec)
      break;
    received += n;
  }
  ASIO_CHECK(received == 2);
  ASIO_CHECK(sizes[0] == 10);
  ASIO_CHECK(sizes[1] == 4);

  c2.send(buffer(data, 100));
  sizes.fill(0);
  out_flags.fill(0);
  std::size_t n = retry(
      [&](asio::error_code& e)
      {
        return c1.receive_multiple(make_span(small_bufs.data(), 1),
            make_span(sizes.data(), 1), 0,
            make_span(out_flags.data(), 1), e);
      }, ec);
  ASIO_CHECK(!ec);
  ASIO_CHECK(n == 1);
  ASIO_CHECK(sizes[0] == 10);

  // A single datagram, without flags.

  std::array<const_buffer, 1> one_buf = { { buffer(data, 42) } };
  n = c2.send_multiple(one_buf, 0, ec);
  ASIO_CHECK(!ec);
  ASIO_CHECK(n == 1);
  sizes.fill(0);
  n = retry(
      [&](asio::error_code& e)
      {
        return c1.receive_multiple(make_span(recv_bufs.data(), 1),
            make_span(sizes.data(), 1), 0, e);
      }, ec);
  ASIO_CHECK(!ec);
  ASIO_CHECK(n == 1);
  ASIO_CHECK(sizes[0] == 42);
  ASIO_CHECK(memcmp(recv_data[0], data, 42) == 0);

  // Ranges of different lengths.

  for (std::size_t i = 0; i < 3; ++i)
    a.send_to(buffer(data, 11 + i), r_ep);

  sizes.fill(0);
  n = retry(
      [&](asio::error_code& e)
      {
        return r.receive_from_multiple(std::ref(recv_bufs),
            std::ref(endpoints), make_span(sizes.data(), 2), 0, e);
      }, ec);
  ASIO_CHECK(!ec);
  ASIO_CHECK(n >= 1 && n <= 2);
  ASIO_CHECK(sizes[2] == 0);
  received = n;
  while (received < 3)
  {
    n = retry(
        [&](asio::error_code& e)
        {
          return r.receive_from_multiple(std::ref(recv_bufs),
              std::ref(endpoints), make_span(sizes.data(), 1), 0, e);
        }, ec);
    ASIO_CHECK(!ec);
    ASIO_CHECK(n == 1);
    if (ec)
      break;
    received += n;
  }
  ASIO_CHECK(received == 3);
  ASIO_CHECK(sizes[0] == 13);

  // Empty ranges.

  std::array<const_buffer, 0> no_send_bufs = { {} };
  std::array<mutable_buffer, 0> no_recv_bufs = { {} };
  std::array<std::size_t, 0> no_sizes = { {} };
  n = c2.send_multiple(no_send_bufs, 0, ec);
  ASIO_CHECK(!ec);
  ASIO_CHECK(n == 0);
  n = c1.receive_multiple(no_recv_bufs, std::ref(no_sizes), 0, ec);
  ASIO_CHECK(!ec);
  ASIO_CHECK(n == 0);

  async_result empty_send, empty_recv;
  c2.async_send_multiple(no_send_bufs, async_handler(&empty_send));
  c1.async_receive_multiple(no_recv_bufs, std::ref(no_sizes),
      async_handler(&empty_recv));
  ioc.restart();
  ioc.run_for(std::chrono::seconds(10));
  ASIO_CHECK(empty_send.called);
  ASIO_CHECK(!empty_send.ec);
  ASIO_CHECK(empty_send.n == 0);
  ASIO_CHECK(empty_recv.called);
  ASIO_CHECK(!empty_recv.ec);
  ASIO_CHECK(empty_recv.n == 0);

  // Repeated destination.

  std::array<std::reference_wrapper<const ip::udp::endpoint>, 3>
    repeated = { { std::cref(r_ep), std::cref(r_ep), std::cref(r_ep) } };
  sent = 0;
  while (sent < send_bufs.size())
  {
    n = a.send_to_multiple(
        make_span(send_bufs.data() + sent, send_bufs.size() - sent),
        make_span(repeated.data() + sent, repeated.size() - sent), 0, ec);
    ASIO_CHECK(!ec);
    ASIO_CHECK(n > 0);
    if (ec)
      break;
    sent += n;
  }

  std::size_t ref_sizes[3] = { 0, 0, 0 };
  std::array<std::reference_wrapper<std::size_t>, 3> size_refs = { {
      std::ref(ref_sizes[0]), std::ref(ref_sizes[1]),
      std::ref(ref_sizes[2]) } };
  received = 0;
  while (received < 3)
  {
    n = retry(
        [&](asio::error_code& e)
        {
          return r.receive_from_multiple(
              make_span(recv_bufs.data() + received, 3 - received),
              make_span(endpoints.data() + received, 3 - received),
              make_span(size_refs.data() + received, 3 - received), 0, e);
        }, ec);
    ASIO_CHECK(!ec);
    if (ec)
      break;
    received += n;
  }
  ASIO_CHECK(received == 3);
  ASIO_CHECK(ref_sizes[0] == 10);
  ASIO_CHECK(ref_sizes[1] == 20);
  ASIO_CHECK(ref_sizes[2] == 30);
  ASIO_CHECK(endpoints[1].port() == a.local_endpoint().port());

  // Destinations yielded by value.

  sent = 0;
  while (sent < send_bufs.size())
  {
    n = a.send_to_multiple(
        make_span(send_bufs.data() + sent, send_bufs.size() - sent),
        test_value_range(r_ep, send_bufs.size() - sent), 0, ec);
    ASIO_CHECK(!ec);
    ASIO_CHECK(n > 0);
    if (ec)
      break;
    sent += n;
  }

  async_result value_send;
  a.async_send_to_multiple(make_span(send_bufs.data(), 1),
      test_value_range(r_ep, 1), async_handler(&value_send));
  ioc.restart();
  while (!value_send.called && ioc.run_one_for(std::chrono::seconds(10)))
  {
  }
  ASIO_CHECK(value_send.called);
  ASIO_CHECK(!value_send.ec);
  ASIO_CHECK(value_send.n == 1);

  ip::udp::resolver resolver(ioc);
  ip::udp::resolver::results_type results = resolver.resolve(
      r_ep.address().to_string(), std::to_string(r_ep.port()),
      ip::udp::resolver::numeric_host | ip::udp::resolver::numeric_service);
  ASIO_CHECK(results.size() == 1);
  n = a.send_to_multiple(make_span(send_bufs.data(), 1), results, 0, ec);
  ASIO_CHECK(!ec);
  ASIO_CHECK(n == 1);

  received = 0;
  while (received < 5)
  {
    sizes.fill(0);
    n = retry(
        [&](asio::error_code& e)
        {
          return r.receive_from_multiple(
              make_span(recv_bufs.data(), 5 - received),
              make_span(endpoints.data(), 5 - received),
              make_span(sizes.data(), 5 - received), 0, e);
        }, ec);
    ASIO_CHECK(!ec);
    if (ec)
      break;
    for (std::size_t i = 0; i < n; ++i)
    {
      ASIO_CHECK(sizes[i] == (received + i < 3 ? 10 * (received + i + 1) : 10));
      ASIO_CHECK(endpoints[i].port() == a.local_endpoint().port());
    }
    received += n;
  }
  ASIO_CHECK(received == 5);

#if defined(__cpp_lib_ranges_repeat)
  // Unbounded destinations.

  auto unbounded = std::views::repeat(r_ep);
  sent = 0;
  while (sent < send_bufs.size())
  {
    n = a.send_to_multiple(
        make_span(send_bufs.data() + sent, send_bufs.size() - sent),
        unbounded, 0, ec);
    ASIO_CHECK(!ec);
    ASIO_CHECK(n > 0);
    if (ec)
      break;
    sent += n;
  }

  received = 0;
  while (received < 3)
  {
    n = retry(
        [&](asio::error_code& e)
        {
          return r.receive_from_multiple(
              make_span(recv_bufs.data() + received, 3 - received),
              make_span(endpoints.data() + received, 3 - received),
              make_span(sizes.data() + received, 3 - received), 0, e);
        }, ec);
    ASIO_CHECK(!ec);
    if (ec)
      break;
    received += n;
  }
  ASIO_CHECK(received == 3);
  ASIO_CHECK(sizes[0] == 10);
  ASIO_CHECK(sizes[1] == 20);
  ASIO_CHECK(sizes[2] == 30);
#endif // defined(__cpp_lib_ranges_repeat)

  // Asynchronous receive before data arrives.

  socket_base::message_flags flag_values[3] = { -1, -1, -1 };
  std::array<std::reference_wrapper<socket_base::message_flags>, 3>
    flag_refs = { { std::ref(flag_values[0]),
      std::ref(flag_values[1]), std::ref(flag_values[2]) } };

  sizes.fill(0);
  async_result recv1;
  r.async_receive_from_multiple(std::ref(recv_bufs), std::ref(endpoints),
      make_span(sizes.data(), 8), make_span(flag_refs.data(), 3),
      async_handler(&recv1));

  async_result send1;
  c2.async_send_multiple(send_bufs, async_handler(&send1));
  a.send_to(buffer(data, 9), r_ep);

  ioc.restart();
  while (!(recv1.called && send1.called)
      && ioc.run_one_for(std::chrono::seconds(10)))
  {
  }
  ASIO_CHECK(send1.called);
  ASIO_CHECK(!send1.ec);
  ASIO_CHECK(send1.n >= 1 && send1.n <= 3);
  ASIO_CHECK(recv1.called);
  ASIO_CHECK(!recv1.ec);
  ASIO_CHECK(recv1.n == 1);
  ASIO_CHECK(sizes[0] == 9);
  ASIO_CHECK(endpoints[0].port() == a.local_endpoint().port());
  ASIO_CHECK(flag_values[0] == 0);
  ASIO_CHECK(flag_values[1] == -1);

  // Deferred receive after data arrives.

  received = 0;
  std::size_t c1_expected = send1.n;
  while (received < c1_expected)
  {
    sizes.fill(0);
    async_result recv2;
    auto op = c1.async_receive_multiple(
        make_span(recv_bufs.data(), 8), make_span(sizes.data(), 8),
        asio::deferred);
    std::move(op)(async_handler(&recv2));
    ioc.restart();
    while (!recv2.called && ioc.run_one_for(std::chrono::seconds(10)))
    {
    }
    ASIO_CHECK(recv2.called);
    ASIO_CHECK(!recv2.ec);
    if (!recv2.called || recv2.ec)
      break;
    ASIO_CHECK(sizes[0] == 10 + 10 * received);
    received += recv2.n;
  }
  ASIO_CHECK(received == c1_expected);
}

} // namespace ip_udp_socket_multiple_runtime

//------------------------------------------------------------------------------

// ip_udp_resolver_compile test
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The following test checks that all public member functions on the class
// ip::udp::resolver compile and link correctly. Runtime failures are ignored.

namespace ip_udp_resolver_compile {

struct resolve_handler
{
  resolve_handler() {}
  void operator()(const asio::error_code&,
      asio::ip::udp::resolver::results_type) {}
  resolve_handler(resolve_handler&&) {}
private:
  resolve_handler(const resolve_handler&);
};

void test()
{
  using namespace asio;
  namespace ip = asio::ip;

  try
  {
    io_context ioc;
    const io_context::executor_type ioc_ex = ioc.get_executor();
    archetypes::lazy_handler lazy;
    asio::error_code ec;
    ip::udp::endpoint e(ip::address_v4::loopback(), 0);

    // basic_resolver constructors.

    ip::udp::resolver resolver(ioc);
    ip::udp::resolver resolver2(ioc_ex);

    ip::udp::resolver resolver3(std::move(resolver));

    // basic_resolver operators.

    resolver = ip::udp::resolver(ioc);
    resolver = std::move(resolver3);

    // I/O object functions.

    ip::udp::resolver::executor_type ex = resolver.get_executor();
    (void)ex;

    // basic_resolver functions.

    resolver.cancel();

    ip::udp::resolver::results_type results1 = resolver.resolve("", "");
    (void)results1;

    ip::udp::resolver::results_type results2 = resolver.resolve("", "", ec);
    (void)results2;

    ip::udp::resolver::results_type results3 =
      resolver.resolve("", "", ip::udp::resolver::flags());
    (void)results3;

    ip::udp::resolver::results_type results4 =
      resolver.resolve("", "", ip::udp::resolver::flags(), ec);
    (void)results4;

    ip::udp::resolver::results_type results5 =
      resolver.resolve(ip::udp::v4(), "", "");
    (void)results5;

    ip::udp::resolver::results_type results6 =
      resolver.resolve(ip::udp::v4(), "", "", ec);
    (void)results6;

    ip::udp::resolver::results_type results7 =
      resolver.resolve(ip::udp::v4(), "", "", ip::udp::resolver::flags());
    (void)results7;

    ip::udp::resolver::results_type results8 =
      resolver.resolve(ip::udp::v4(), "", "", ip::udp::resolver::flags(), ec);
    (void)results8;

    ip::udp::resolver::results_type results9 = resolver.resolve(e);
    (void)results9;

    ip::udp::resolver::results_type results10 = resolver.resolve(e, ec);
    (void)results10;

    resolver.async_resolve("", "", resolve_handler());
    int i1 = resolver.async_resolve("", "", lazy);
    (void)i1;

    resolver.async_resolve("", "",
        ip::udp::resolver::flags(), resolve_handler());
    int i2 = resolver.async_resolve("", "",
        ip::udp::resolver::flags(), lazy);
    (void)i2;

    resolver.async_resolve(ip::udp::v4(), "", "", resolve_handler());
    int i3 = resolver.async_resolve(ip::udp::v4(), "", "", lazy);
    (void)i3;

    resolver.async_resolve(ip::udp::v4(),
        "", "", ip::udp::resolver::flags(), resolve_handler());
    int i4 = resolver.async_resolve(ip::udp::v4(),
        "", "", ip::udp::resolver::flags(), lazy);
    (void)i4;

    resolver.async_resolve(e, resolve_handler());
    int i5 = resolver.async_resolve(e, lazy);
    (void)i5;
  }
  catch (std::exception&)
  {
  }
}

} // namespace ip_udp_resolver_compile

//------------------------------------------------------------------------------

ASIO_TEST_SUITE
(
  "ip/udp",
  ASIO_COMPILE_TEST_CASE(ip_udp_socket_compile::test)
  ASIO_TEST_CASE(ip_udp_socket_runtime::test)
  ASIO_TEST_CASE(ip_udp_socket_multiple_runtime::test)
  ASIO_COMPILE_TEST_CASE(ip_udp_resolver_compile::test)
)
