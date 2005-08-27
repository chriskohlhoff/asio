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
#include "asio/ipv4/socket_option.hpp"

namespace asio {
namespace ipv4 {
namespace multicast {

namespace socket_option {

/// Helper template for implementing ip_mreq-based options.
template <int Level, int Name>
class request
{
public:
  /// Default constructor.
  request()
  {
    value_.imr_multiaddr.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          asio::ipv4::address::any().to_ulong());
    value_.imr_interface.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          asio::ipv4::address::any().to_ulong());
  }

  /// Construct with multicast address only.
  request(const asio::ipv4::address& multicast_address)
  {
    value_.imr_multiaddr.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          multicast_address.to_ulong());
    value_.imr_interface.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          asio::ipv4::address::any().to_ulong());
  }

  /// Construct with multicast address and address of local interface to use.
  request(const asio::ipv4::address& multicast_address,
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

  /// Get the address of the option data.
  void* data()
  {
    return &value_;
  }

  /// Get the address of the option data.
  const void* data() const
  {
    return &value_;
  }

  /// Get the size of the option data.
  size_t size() const
  {
    return sizeof(value_);
  }

private:
  ip_mreq value_;
};

} // namespace socket_option

/// Socket option to join a multicast group on a specified interface.
typedef asio::ipv4::multicast::socket_option::request<
  IPPROTO_IP, IP_ADD_MEMBERSHIP> add_membership;

/// Socket option to leave a multicast group on a specified interface.
typedef asio::ipv4::multicast::socket_option::request<
  IPPROTO_IP, IP_DROP_MEMBERSHIP> drop_membership;

/// Socket option for local interface to use for outgoing multicast packets.
typedef asio::ipv4::socket_option::address<
  IPPROTO_IP, IP_MULTICAST_IF> outbound_interface;

/// Socket option for time-to-live associated with outgoing multicast packets.
typedef asio::socket_option::integer<
  IPPROTO_IP, IP_MULTICAST_TTL> time_to_live;

/// Socket option determining whether outgoing multicast packets will be
/// received on the same socket if it is a member of the multicast group.
typedef asio::socket_option::boolean<
  IPPROTO_IP, IP_MULTICAST_LOOP> enable_loopback;

} // namespace multicast
} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_MULTICAST_HPP
