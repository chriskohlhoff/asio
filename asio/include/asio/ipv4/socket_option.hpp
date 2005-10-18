//
// socket_option.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_SOCKET_OPTION_HPP
#define ASIO_IPV4_SOCKET_OPTION_HPP

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
namespace socket_option {

/// Helper template for implementing address-based options.
template <int Level, int Name>
class address
{
public:
  /// Default constructor.
  address()
  {
    value_.s_addr = asio::detail::socket_ops::host_to_network_long(
        asio::ipv4::address::any().to_ulong());
  }

  /// Construct with address.
  address(const asio::ipv4::address& value)
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
#if defined(GENERATING_DOCUMENTATION)
  implementation_defined data()
#else
  in_addr* data()
#endif
  {
    return &value_;
  }

  /// Get the address of the flag data.
#if defined(GENERATING_DOCUMENTATION)
  implementation_defined data() const
#else
  const in_addr* data() const
#endif
  {
    return &value_;
  }

  /// Get the size of the flag data.
  std::size_t size() const
  {
    return sizeof(value_);
  }

private:
  in_addr value_;
};

} // namespace socket_option
} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_SOCKET_OPTION_HPP
