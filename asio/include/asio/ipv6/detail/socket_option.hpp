//
// socket_option.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV6_DETAIL_SOCKET_OPTION_HPP
#define ASIO_IPV6_DETAIL_SOCKET_OPTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <cstring>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/ipv6/address.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv6 {
namespace detail {
namespace socket_option {

// Helper template for implementing ip_mreq-based options.
template <int Level, int Name>
class multicast_request
{
public:
  // Default constructor.
  multicast_request()
  {
    in6_addr tmp_addr = IN6ADDR_ANY_INIT;
    value_.ipv6mr_multiaddr = tmp_addr;
    value_.ipv6mr_interface = 0;
  }

  // Construct with multicast address only.
  multicast_request(const asio::ipv6::address& multicast_address)
  {
    using namespace std; // For memcpy.
    asio::ipv6::address::bytes_type bytes = multicast_address.to_bytes();
    memcpy(value_.ipv6mr_multiaddr.s6_addr, bytes.elems, 16);
    value_.ipv6mr_interface = 0;
  }

  // Construct with multicast address and address of local interface to use.
  multicast_request(const asio::ipv6::address& multicast_address,
      unsigned int local_interface)
  {
    using namespace std; // For memcpy.
    asio::ipv6::address::bytes_type bytes = multicast_address.to_bytes();
    memcpy(value_.ipv6mr_multiaddr.s6_addr, bytes.elems, 16);
    value_.ipv6mr_interface = local_interface;
  }

  // Get the level of the socket option.
  int level() const
  {
    return Level;
  }

  // Get the name of the socket option.
  int name() const
  {
    return Name;
  }

  // Get the address of the option data.
  ipv6_mreq* data()
  {
    return &value_;
  }

  // Get the address of the option data.
  const ipv6_mreq* data() const
  {
    return &value_;
  }

  // Get the size of the option data.
  std::size_t size() const
  {
    return sizeof(value_);
  }

private:
  ipv6_mreq value_;
};

} // namespace socket_option
} // namespace detail
} // namespace ipv6
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV6_DETAIL_SOCKET_OPTION_HPP
