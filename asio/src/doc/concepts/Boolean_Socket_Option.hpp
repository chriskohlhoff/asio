//
// Boolean_Socket_Option.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Boolean_Socket_Option concept.
/**
 * @par Implemented By:
 * asio::socket_base::broadcast @n
 * asio::socket_base::do_not_route @n
 * asio::socket_base::keep_alive @n
 * asio::socket_base::reuse_address @n
 * asio::ipv4::tcp::no_delay @n
 * asio::ipv4::multicast::enable_loopback
 */
class Boolean_Socket_Option
  : public Socket_Option
{
public:
  /// Default constructor initialises contained value to 0.
  Boolean_Socket_Option();

  /// Construct with a specific option value.
  Boolean_Socket_Option(bool value);

  /// Set the value of the boolean option.
  void set(bool value);

  /// Get the current value of the boolean option.
  bool get() const;
};
