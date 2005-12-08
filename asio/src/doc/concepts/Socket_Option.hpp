//
// Socket_Option.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Socket_Option concept.
/**
 * @par Implemented By:
 * asio::socket_base::broadcast @n
 * asio::socket_base::do_not_route @n
 * asio::socket_base::keep_alive @n
 * asio::socket_base::linger @n
 * asio::socket_base::send_buffer_size @n
 * asio::socket_base::send_low_watermark @n
 * asio::socket_base::receive_buffer_size @n
 * asio::socket_base::receive_low_watermark @n
 * asio::socket_base::reuse_address @n
 * asio::ipv4::tcp::no_delay @n
 * asio::ipv4::multicast::add_membership @n
 * asio::ipv4::multicast::drop_membership @n
 * asio::ipv4::multicast::outbound_interface @n
 * asio::ipv4::multicast::time_to_live @n
 * asio::ipv4::multicast::enable_loopback
 */
class Socket_Option
{
public:
  /// Get the level of the socket option.
  int level() const;

  /// Get the name of the socket option.
  int name() const;

  /// Get a pointer to the socket option data.
  implementation_defined data();

  /// Get a pointer to the socket option data.
  implementation_defined data() const;

  /// Get the size of the socket option data in bytes.
  std::size_t size() const;
};
