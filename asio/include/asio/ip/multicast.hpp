//
// multicast.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_MULTICAST_HPP
#define ASIO_IP_MULTICAST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/ip/detail/socket_option.hpp"

namespace asio {
namespace ip {
namespace multicast {

/// Socket option to join a multicast group on a specified interface.
/**
 * Implements the IPPROTO_IP/IP_ADD_MEMBERSHIP socket option.
 *
 * @par Examples:
 * Setting the option to join a multicast group:
 * @code
 * asio::ip::udp::socket socket(io_service); 
 * ...
 * asio::ip::address multicast_address("225.0.0.1");
 * asio::ip::multicast::join_group option(multicast_address);
 * socket.set_option(option);
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, IP_MReq_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined join_group;
#else
typedef asio::ip::detail::socket_option::multicast_request<
  IPPROTO_IP, IP_ADD_MEMBERSHIP, IPPROTO_IPV6, IPV6_JOIN_GROUP> join_group;
#endif

/// Socket option to leave a multicast group on a specified interface.
/**
 * Implements the IPPROTO_IP/IP_DROP_MEMBERSHIP socket option.
 *
 * @par Examples:
 * Setting the option to leave a multicast group:
 * @code
 * asio::ip::udp::socket socket(io_service); 
 * ...
 * asio::ip::address multicast_address("225.0.0.1");
 * asio::ip::multicast::leave_group option(multicast_address);
 * socket.set_option(option);
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, IP_MReq_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined leave_group;
#else
typedef asio::ip::detail::socket_option::multicast_request<
  IPPROTO_IP, IP_DROP_MEMBERSHIP, IPPROTO_IPV6, IPV6_LEAVE_GROUP> leave_group;
#endif

/// Socket option for time-to-live associated with outgoing multicast packets.
/**
 * Implements the IPPROTO_IP/IP_MULTICAST_TTL socket option.
 *
 * @par Examples:
 * Setting the option:
 * @code
 * asio::ip::udp::socket socket(io_service); 
 * ...
 * asio::ip::multicast::hops option(4);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Getting the current option value:
 * @code
 * asio::ip::udp::socket socket(io_service); 
 * ...
 * asio::ip::multicast::hops option;
 * socket.get_option(option);
 * int ttl = option.get();
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, Integer_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined hops;
#else
typedef asio::ip::detail::socket_option::integer<
  IPPROTO_IP, IP_MULTICAST_TTL, IPPROTO_IPV6, IPV6_MULTICAST_HOPS> hops;
#endif

/// Socket option determining whether outgoing multicast packets will be
/// received on the same socket if it is a member of the multicast group.
/**
 * Implements the IPPROTO_IP/IP_MULTICAST_LOOP socket option.
 *
 * @par Examples:
 * Setting the option:
 * @code
 * asio::ip::udp::socket socket(io_service); 
 * ...
 * asio::ip::multicast::enable_loopback option(true);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Getting the current option value:
 * @code
 * asio::ip::udp::socket socket(io_service); 
 * ...
 * asio::ip::multicast::enable_loopback option;
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
typedef asio::ip::detail::socket_option::boolean<
  IPPROTO_IP, IP_MULTICAST_LOOP, IPPROTO_IPV6, IPV6_MULTICAST_LOOP>
  enable_loopback;
#endif

} // namespace multicast
} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_MULTICAST_HPP
