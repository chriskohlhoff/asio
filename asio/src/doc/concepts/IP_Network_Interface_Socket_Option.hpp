//
// IP_Network_Interface_Socket_Option.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// IP_Network_Interface_Socket_Option concept.
/**
 * @par Implemented By:
 * asio::ip::multicast::outbound_interface
 */
class IP_Network_Interface_Socket_Option
  : public Socket_Option
{
public:
  /// Default constructor initialises contained value to "any" interface.
  IP_Network_Interface_Socket_Option();

  /// Construct with a specific IPv4 address.
  IP_Network_Interface_Socket_Option(
      const asio::ip::address_v4& ipv4_interface);

  /// Construct with a specific IPv6 interface index.
  IP_Network_Interface_Socket_Option(
      unsigned long ipv6_interface);
};
