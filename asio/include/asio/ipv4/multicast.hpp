//
// multicast.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_MULTICAST_HPP
#define ASIO_IPV4_MULTICAST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/socket_option.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv4 {
namespace multicast {

/// Helper template for implementing ip_mreq-based options.
template <int Level, int Name>
class ip_mreq_option
{
public:
  /// Default constructor.
  ip_mreq_option()
  {
    value_.imr_multiaddr.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          asio::ipv4::address::any().to_ulong());
    value_.imr_interface.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          asio::ipv4::address::any().to_ulong());
  }

  /// Construct with multicast address only.
  ip_mreq_option(const asio::ipv4::address& multicast_address)
  {
    value_.imr_multiaddr.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          multicast_address.to_ulong());
    value_.imr_interface.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          asio::ipv4::address::any().to_ulong());
  }

  /// Construct with multicast address and address of local interface to use.
  ip_mreq_option(const asio::ipv4::address& multicast_address,
      const asio::ipv4::address& local_address)
  {
    value_.imr_multiaddr.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          multicast_address.to_ulong());
    value_.imr_interface.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          local_address.to_ulong());
  }

  /// Get the level of the socket option.
  int level() const
  {
    return Level;
  }

  /// Get the name of the socket option.
  int name() const
  {
    return Name;
  }

  /// Get the address of the flag data.
  void* data()
  {
    return &value_;
  }

  /// Get the address of the flag data.
  const void* data() const
  {
    return &value_;
  }

  /// Get the size of the flag data.
  size_t size() const
  {
    return sizeof(value_);
  }

private:
  ip_mreq value_;
};

/// Helper template for implementing address-based options.
template <int Level, int Name>
class address_option
{
public:
  /// Default constructor.
  address_option()
  {
    value_.s_addr = asio::detail::socket_ops::host_to_network_long(
        asio::ipv4::address::any().to_ulong());
  }

  /// Construct with address.
  address_option(const asio::ipv4::address& value)
  {
    value_.s_addr =
      asio::detail::socket_ops::host_to_network_long(value.to_ulong());
  }

  /// Get the level of the socket option.
  int level() const
  {
    return Level;
  }

  /// Get the name of the socket option.
  int name() const
  {
    return Name;
  }

  /// Set the value of the socket option.
  void set(const asio::ipv4::address& value)
  {
    value_.s_addr =
      asio::detail::socket_ops::host_to_network_long(value.to_ulong());
  }

  /// Get the current value of the socket option.
  asio::ipv4::address get() const
  {
    return asio::ipv4::address(
        asio::detail::socket_ops::network_to_host_long(value_.s_addr));
  }

  /// Get the address of the flag data.
  void* data()
  {
    return &value_;
  }

  /// Get the address of the flag data.
  const void* data() const
  {
    return &value_;
  }

  /// Get the size of the flag data.
  size_t size() const
  {
    return sizeof(value_);
  }

private:
  in_addr value_;
};

/// Join a multicast group on a specified interface.
typedef ip_mreq_option<IPPROTO_IP, IP_ADD_MEMBERSHIP> add_membership;

/// Leave a multicast group on a specified interface.
typedef ip_mreq_option<IPPROTO_IP, IP_DROP_MEMBERSHIP> drop_membership;

/// Local interface to use for outgoing multicast packets.
typedef address_option<IPPROTO_IP, IP_MULTICAST_IF> outbound_interface;

/// Time-to-live associated with outgoing multicast packets.
typedef socket_option::integer<IPPROTO_IP, IP_MULTICAST_TTL> time_to_live;

/// Whether outgoing multicast packets will be received on the same socket if
/// it is a member of the multicast group.
typedef socket_option::flag<IPPROTO_IP, IP_MULTICAST_LOOP> enable_loopback;

} // namespace multicast
} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_MULTICAST_HPP
