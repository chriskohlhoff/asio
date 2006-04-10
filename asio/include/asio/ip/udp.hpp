//
// udp.hpp
// ~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_UDP_HPP
#define ASIO_IP_UDP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_datagram_socket.hpp"
#include "asio/datagram_socket_service.hpp"
#include "asio/ip/basic_endpoint.hpp"
#include "asio/ipv4/udp.hpp"
#include "asio/ipv6/udp.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ip {

/// Encapsulates the flags needed for UDP.
/**
 * The asio::ip::udp class contains flags necessary for UDP sockets.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe.
 *
 * @par Concepts:
 * Protocol.
 */
class udp
{
public:
  /// The IPv4 protocol type.
  typedef asio::ipv4::udp ipv4_protocol;

  /// The IPv6 protocol type.
  typedef asio::ipv6::udp ipv6_protocol;

  /// The type of a UDP endpoint.
  typedef basic_endpoint<udp> endpoint;

  /// Construct to represent the IPv4 UDP protocol.
  udp(const ipv4_protocol&)
    : family_(PF_INET)
  {
  }

  /// Construct to represent the IPv4 UDP protocol.
  udp(const ipv6_protocol&)
    : family_(PF_INET6)
  {
  }

  /// Obtain an identifier for the type of the protocol.
  int type() const
  {
    return SOCK_DGRAM;
  }

  /// Obtain an identifier for the protocol.
  int protocol() const
  {
    return IPPROTO_UDP;
  }

  /// Obtain an identifier for the protocol family.
  int family() const
  {
    return family_;
  }

  /// The service type for IPv4 UDP sockets.
  typedef datagram_socket_service<udp> socket_service;

  /// The IPv4 UDP socket type.
  typedef basic_datagram_socket<socket_service> socket;

private:
  int family_;
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_UDP_HPP
