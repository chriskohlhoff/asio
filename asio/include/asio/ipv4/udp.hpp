//
// udp.hpp
// ~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_UDP_HPP
#define ASIO_IPV4_UDP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/ipv4/address.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv4 {

/// Encapsulates the flags needed for UDP.
/**
 * The asio::ipv4::udp class contains flags necessary for UDP sockets.
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
  class endpoint;

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
    return PF_INET;
  }
};

/// Describes an endpoint for a UDP socket.
/**
 * The asio::ipv4::udp::endpoint class describes an endpoint that may be
 * associated with a particular socket.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Endpoint.
 */
class udp::endpoint
{
public:
  /// The protocol type associated with the endpoint.
  typedef udp protocol_type;

  /// The type of the endpoint structure. This type is dependent on the
  /// underlying implementation of the socket layer.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined data_type;
#else
  typedef asio::detail::socket_addr_type data_type;
#endif

  /// The type for the size of the endpoint structure. This type is dependent on
  /// the underlying implementation of the socket layer.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined size_type;
#else
  typedef asio::detail::socket_addr_len_type size_type;
#endif

  /// Default constructor.
  endpoint()
  {
    addr_.sin_family = AF_INET;
    addr_.sin_port = 0;
    addr_.sin_addr.s_addr = INADDR_ANY;
  }

  /// Construct an endpoint using a port number, specified in the host's byte
  /// order. The IP address will be the any address (i.e. INADDR_ANY). This
  /// constructor would typically be used for accepting new connections.
  endpoint(unsigned short port_num)
  {
    addr_.sin_family = AF_INET;
    addr_.sin_port =
      asio::detail::socket_ops::host_to_network_short(port_num);
    addr_.sin_addr.s_addr = INADDR_ANY;
  }

  /// Construct an endpoint using a port number and an IP address. This
  /// constructor may be used for accepting connections on a specific interface
  /// or for making a connection to a remote endpoint.
  endpoint(unsigned short port_num, const asio::ipv4::address& addr)
  {
    addr_.sin_family = AF_INET;
    addr_.sin_port =
      asio::detail::socket_ops::host_to_network_short(port_num);
    addr_.sin_addr.s_addr =
      asio::detail::socket_ops::host_to_network_long(addr.to_ulong());
  }

  /// Copy constructor.
  endpoint(const endpoint& other)
    : addr_(other.addr_)
  {
  }

  /// Assign from another endpoint.
  endpoint& operator=(const endpoint& other)
  {
    addr_ = other.addr_;
    return *this;
  }

  /// The protocol associated with the endpoint.
  protocol_type protocol() const
  {
    return protocol_type();
  }

  /// Get the underlying endpoint in the native type.
  data_type* data()
  {
    return reinterpret_cast<data_type*>(&addr_);
  }

  /// Get the underlying endpoint in the native type.
  const data_type* data() const
  {
    return reinterpret_cast<const data_type*>(&addr_);
  }

  /// Get the underlying size of the endpoint in the native type.
  size_type size() const
  {
    return sizeof(addr_);
  }

  /// Set the underlying size of the endpoint in the native type.
  void size(size_type size)
  {
    if (size != sizeof(addr_))
    {
      asio::error e(asio::error::invalid_argument);
      boost::throw_exception(e);
    }
  }

  /// Get the port associated with the endpoint. The port number is always in
  /// the host's byte order.
  unsigned short port() const
  {
    return asio::detail::socket_ops::network_to_host_short(
        addr_.sin_port);
  }

  /// Set the port associated with the endpoint. The port number is always in
  /// the host's byte order.
  void port(unsigned short port_num)
  {
    addr_.sin_port =
      asio::detail::socket_ops::host_to_network_short(port_num);
  }

  /// Get the IP address associated with the endpoint.
  asio::ipv4::address address() const
  {
    return asio::detail::socket_ops::network_to_host_long(
        addr_.sin_addr.s_addr);
  }

  /// Set the IP address associated with the endpoint.
  void address(const asio::ipv4::address& addr)
  {
    addr_.sin_addr.s_addr =
      asio::detail::socket_ops::host_to_network_long(addr.to_ulong());
  }

  /// Compare two endpoints for equality.
  friend bool operator==(const endpoint& e1, const endpoint& e2)
  {
    return e1.address() == e2.address() && e1.port() == e2.port();
  }

  /// Compare two endpoints for inequality.
  friend bool operator!=(const endpoint& e1, const endpoint& e2)
  {
    return e1.address() != e2.address() || e1.port() != e2.port();
  }

  /// Compare endpoints for ordering.
  friend bool operator<(const endpoint& e1, const endpoint& e2)
  {
    if (e1.address() < e2.address())
      return true;
    if (e1.address() != e2.address())
      return false;
    return e1.port() < e2.port();
  }

private:
  // The underlying IPv4 socket address.
  asio::detail::inet_addr_v4_type addr_;
};

/// Output an endpoint as a string.
/**
 * Used to output a human-readable string for a specified endpoint.
 *
 * @param os The output stream to which the string will be written.
 *
 * @param endpoint The endpoint to be written.
 *
 * @return The output stream.
 *
 * @relates tcp::endpoint
 */
template <typename Ostream>
Ostream& operator<<(Ostream& os, const udp::endpoint& endpoint)
{
  os << endpoint.address().to_string() << ':' << endpoint.port();
  return os;
}

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_UDP_HPP
