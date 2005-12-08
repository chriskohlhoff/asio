//
// Integer_Socket_Option.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Integer_Socket_Option concept.
/**
 * @par Implemented By:
 * asio::socket_base::send_buffer_size @n
 * asio::socket_base::send_low_watermark @n
 * asio::socket_base::receive_buffer_size @n
 * asio::socket_base::receive_low_watermark @n
 * asio::ipv4::multicast::time_to_live
 */
class Integer_Socket_Option
  : public Socket_Option
{
public:
  /// Default constructor initialises contained value to 0.
  Integer_Socket_Option();

  /// Construct with a specific option value.
  Integer_Socket_Option(int value);

  /// Set the value of the integer option.
  void set(int value);

  /// Get the current value of the integer option.
  int get() const;
};
