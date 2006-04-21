//
// socket_option.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_DETAIL_SOCKET_OPTION_HPP
#define ASIO_IPV4_DETAIL_SOCKET_OPTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/ipv4/address.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv4 {
namespace detail {
namespace socket_option {

// Helper template for implementing address-based options.
template <int Level, int Name>
class address
{
public:
  // Default constructor.
  address()
  {
    value_.s_addr = asio::detail::socket_ops::host_to_network_long(
        asio::ipv4::address::any().to_ulong());
  }

  // Construct with address.
  address(const asio::ipv4::address& value)
  {
    value_.s_addr =
      asio::detail::socket_ops::host_to_network_long(value.to_ulong());
  }

  // Set the value of the socket option.
  void set(const asio::ipv4::address& value)
  {
    value_.s_addr =
      asio::detail::socket_ops::host_to_network_long(value.to_ulong());
  }

  // Get the current value of the socket option.
  asio::ipv4::address get() const
  {
    return asio::ipv4::address(
        asio::detail::socket_ops::network_to_host_long(value_.s_addr));
  }

  // Get the level of the socket option.
  template <typename Protocol>
  int level(const Protocol&) const
  {
    return Level;
  }

  // Get the name of the socket option.
  template <typename Protocol>
  int name(const Protocol&) const
  {
    return Name;
  }

  // Get the address of the option data.
  template <typename Protocol>
  in_addr* data(const Protocol&)
  {
    return &value_;
  }

  // Get the address of the option data.
  template <typename Protocol>
  const in_addr* data(const Protocol&) const
  {
    return &value_;
  }

  // Get the size of the option data.
  template <typename Protocol>
  std::size_t size(const Protocol&) const
  {
    return sizeof(value_);
  }

private:
  in_addr value_;
};

// Helper template for implementing ip_mreq-based options.
template <int Level, int Name>
class multicast_request
{
public:
  // Default constructor.
  multicast_request()
  {
    value_.imr_multiaddr.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          asio::ipv4::address::any().to_ulong());
    value_.imr_interface.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          asio::ipv4::address::any().to_ulong());
  }

  // Construct with multicast address only.
  multicast_request(const asio::ipv4::address& multicast_address)
  {
    value_.imr_multiaddr.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          multicast_address.to_ulong());
    value_.imr_interface.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          asio::ipv4::address::any().to_ulong());
  }

  // Construct with multicast address and address of local interface to use.
  multicast_request(const asio::ipv4::address& multicast_address,
      const asio::ipv4::address& local_address)
  {
    value_.imr_multiaddr.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          multicast_address.to_ulong());
    value_.imr_interface.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          local_address.to_ulong());
  }

  // Get the level of the socket option.
  template <typename Protocol>
  int level(const Protocol&) const
  {
    return Level;
  }

  // Get the name of the socket option.
  template <typename Protocol>
  int name(const Protocol&) const
  {
    return Name;
  }

  // Get the address of the option data.
  template <typename Protocol>
  ip_mreq* data(const Protocol&)
  {
    return &value_;
  }

  // Get the address of the option data.
  template <typename Protocol>
  const ip_mreq* data(const Protocol&) const
  {
    return &value_;
  }

  // Get the size of the option data.
  template <typename Protocol>
  std::size_t size(const Protocol&) const
  {
    return sizeof(value_);
  }

private:
  ip_mreq value_;
};

} // namespace socket_option
} // namespace detail
} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_DETAIL_SOCKET_OPTION_HPP
