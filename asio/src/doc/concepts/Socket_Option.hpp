//
// Socket_Option.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Socket_Option concept.
/**
 * Defines the interface that must be implemented by an object passed as the
 * @c option parameter to:
 * @li asio::tcp::socket::get_option
 * @li asio::tcp::socket::set_option
 * @li asio::udp::socket::get_option
 * @li asio::udp::socket::set_option
 * @li asio::tcp::acceptor::get_option
 * @li asio::tcp::acceptor::set_option
 *
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
 * asio::ip::tcp::no_delay @n
 * asio::ip::multicast::join_group @n
 * asio::ip::multicast::leave_group @n
 * asio::ip::multicast::outbound_interface @n
 * asio::ip::multicast::hops @n
 * asio::ip::multicast::enable_loopback
 */
class Socket_Option
{
public:
  /// Get the level of the socket option.
  template <typename Protocol>
  int level(const Protocol& protocol) const;

  /// Get the name of the socket option.
  template <typename Protocol>
  int name(const Protocol& protocol) const;

  /// Get a pointer to the socket option data.
  template <typename Protocol>
  implementation_defined data(const Protocol& protocol);

  /// Get a pointer to the socket option data.
  template <typename Protocol>
  implementation_defined data(const Protocol& protocol) const;

  /// Get the size of the socket option data in bytes.
  template <typename Protocol>
  std::size_t size(const Protocol& protocol) const;
};
