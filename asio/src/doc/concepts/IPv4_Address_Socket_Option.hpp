//
// IPv4_Address_Socket_Option.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// IPv4_Address_Socket_Option concept.
/**
 * @par Implemented By:
 * asio::ipv4::multicast::outbound_interface
 */
class IPv4_Address_Socket_Option
  : public Socket_Option
{
public:
  /// Default constructor initialises contained value to 0.
  IPv4_Address_Socket_Option();

  /// Construct with a specific option value.
  IPv4_Address_Socket_Option(const asio::ipv4::address& value);

  /// Set the value of the boolean option.
  void set(const asio::ipv4::address& value);

  /// Get the current value of the boolean option.
  asio::ipv4::address get() const;
};
