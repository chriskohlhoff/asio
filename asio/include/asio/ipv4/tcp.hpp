//
// tcp.hpp
// ~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_TCP_HPP
#define ASIO_IPV4_TCP_HPP

#include "asio/detail/push_options.hpp"

#include "asio/error.hpp"
#include "asio/socket_option.hpp"
#include "asio/ipv4/address.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv4 {

/// Encapsulates the flags needed for TCP.
/**
 * The asio::ipv4::tcp class contains the flags necessary to use TCP sockets.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe.
 *
 * @par Concepts:
 * Protocol.
 */
class tcp
{
public:
  class endpoint;

  /// Obtain an identifier for the type of the protocol.
  int type() const
  {
    return SOCK_STREAM;
  }

  /// Obtain an identifier for the protocol.
  int protocol() const
  {
    return IPPROTO_TCP;
  }

  /// Obtain an identifier for the protocol family.
  int family() const
  {
    return PF_INET;
  }

  /// Socket option for disabling the Nagle algorithm.
  typedef socket_option::flag<IPPROTO_TCP, TCP_NODELAY> no_delay;
};

/// Describes an endpoint for a TCP socket.
/**
 * The asio::ipv4::tcp::endpoint class describes an endpoint that may be
 * associated with a particular socket.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Endpoint.
 */
class tcp::endpoint
{
public:
  /// The protocol type associated with the endpoint.
  typedef tcp protocol_type;

  /// The native type of the endpoint structure. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef asio::detail::socket_addr_type native_data_type;

  /// The native type for the size of the endpoint structure. This type is
  /// dependent on the underlying implementation of the socket layer.
  typedef asio::detail::socket_addr_len_type native_size_type;

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
    addr_.sin_port = asio::detail::socket_ops::host_to_network_short(port_num);
    addr_.sin_addr.s_addr = INADDR_ANY;
  }

  /// Construct an endpoint using a port number and an IP address. This
  /// constructor may be used for accepting connections on a specific interface
  /// or for making a connection to a remote endpoint.
  endpoint(unsigned short port_num, const asio::ipv4::address& addr)
  {
    addr_.sin_family = AF_INET;
    addr_.sin_port = asio::detail::socket_ops::host_to_network_short(port_num);
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
  native_data_type* native_data()
  {
    return reinterpret_cast<native_data_type*>(&addr_);
  }

  /// Get the underlying endpoint in the native type.
  const native_data_type* native_data() const
  {
    return reinterpret_cast<const native_data_type*>(&addr_);
  }

  /// Get the underlying size of the endpoint in the native type.
  native_size_type native_size() const
  {
    return sizeof(addr_);
  }

  /// Set the underlying size of the endpoint in the native type.
  void native_size(native_size_type size)
  {
    if (size != sizeof(addr_))
      throw asio::error(asio::error::invalid_argument);
  }

  /// Get the port associated with the endpoint. The port number is always in
  /// the host's byte order.
  unsigned short port() const
  {
    return asio::detail::socket_ops::network_to_host_short(addr_.sin_port);
  }

  /// Set the port associated with the endpoint. The port number is always in
  /// the host's byte order.
  void port(unsigned short port_num)
  {
    addr_.sin_port = asio::detail::socket_ops::host_to_network_short(port_num);
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

private:
  // The underlying IPv4 socket address.
  asio::detail::inet_addr_v4_type addr_;
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_TCP_HPP
