//
// multicast.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV6_MULTICAST_HPP
#define ASIO_IPV6_MULTICAST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_option.hpp"
#include "asio/ipv6/detail/socket_option.hpp"

namespace asio {
namespace ipv6 {
namespace multicast {

/// Socket option to join a multicast group on a specified interface.
/**
 * Implements the IPPROTO_IPV6/IPV6_JOIN_GROUP socket option.
 *
 * @par Examples:
 * Setting the option to join a multicast group:
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * asio::ipv6::address multicast_address("225.0.0.1");
 * asio::ipv6::multicast::join_group option(multicast_address);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Setting the option to join a multicast group on the specified local
 * interface.
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * asio::ipv6::address multicast_address("225.0.0.1");
 * asio::ipv6::multicast::join_group option(multicast_address, 1);
 * socket.set_option(option);
 * ...
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, IPv6_MReq_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined join_group;
#else
typedef asio::ipv6::detail::socket_option::multicast_request<
  IPPROTO_IPV6, IPV6_JOIN_GROUP> join_group;
#endif

/// Socket option to leave a multicast group on a specified interface.
/**
 * Implements the IPPROTO_IPV6/IPV6_LEAVE_GROUP socket option.
 *
 * @par Examples:
 * Setting the option to leave a multicast group:
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * asio::ipv6::address multicast_address("225.0.0.1");
 * asio::ipv6::multicast::leave_group option(multicast_address);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Setting the option to leave a multicast group on the specified local
 * interface.
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * asio::ipv6::address multicast_address("225.0.0.1");
 * asio::ipv6::multicast::leave_group option(multicast_address, 1);
 * socket.set_option(option);
 * ...
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, IPv6_MReq_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined leave_group;
#else
typedef asio::ipv6::detail::socket_option::multicast_request<
  IPPROTO_IPV6, IPV6_LEAVE_GROUP> leave_group;
#endif

/// Socket option for local interface to use for outgoing multicast packets.
/**
 * Implements the IPPROTO_IPV6/IPV6_MULTICAST_IF socket option.
 *
 * @par Examples:
 * Setting the option:
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * unsigned int local_interface = 1;
 * asio::ipv6::multicast::outbound_interface option(local_interface);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Getting the current option value:
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * asio::ipv6::multicast::outbound_interface option;
 * socket.get_option(option);
 * unsigned int local_interface = option.get();
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, Unsigned_Integer_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined outbound_interface;
#else
typedef asio::detail::socket_option::unsigned_integer<
  IPPROTO_IPV6, IPV6_MULTICAST_IF> outbound_interface;
#endif

/// Socket option for number of hops permitted on outgoing multicast packets.
/**
 * Implements the IPPROTO_IPV6/IPV6_MULTICAST_HOPS socket option.
 *
 * @par Examples:
 * Setting the option:
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * asio::ipv6::multicast::hops option(4);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Getting the current option value:
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * asio::ipv6::multicast::hops option;
 * socket.get_option(option);
 * int ttl = option.get();
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, Unsigned_Integer_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined hops;
#else
typedef asio::detail::socket_option::unsigned_integer<
  IPPROTO_IPV6, IPV6_MULTICAST_HOPS> hops;
#endif

/// Socket option determining whether outgoing multicast packets will be
/// received on the same socket if it is a member of the multicast group.
/**
 * Implements the IPPROTO_IPV6/IPV6_MULTICAST_LOOP socket option.
 *
 * @par Examples:
 * Setting the option:
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * asio::ipv6::multicast::enable_loopback option(true);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Getting the current option value:
 * @code
 * asio::ipv6::udp::socket socket(io_service); 
 * ...
 * asio::ipv6::multicast::enable_loopback option;
 * socket.get_option(option);
 * bool is_set = option.get();
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, Boolean_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined enable_loopback;
#else
typedef asio::detail::socket_option::boolean<
  IPPROTO_IPV6, IPV6_MULTICAST_LOOP> enable_loopback;
#endif

} // namespace multicast
} // namespace ipv6
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV6_MULTICAST_HPP
