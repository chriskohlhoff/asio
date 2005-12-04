//
// multicast.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_option.hpp"
#include "asio/ipv4/detail/socket_option.hpp"

namespace asio {
namespace ipv4 {
namespace multicast {

/// Socket option to join a multicast group on a specified interface.
/**
 * Implements the IPPROTO_IP/IP_ADD_MEMBERSHIP socket option.
 *
 * @par Examples:
 * Setting the option to join a multicast group:
 * @code
 * asio::datagram_socket socket(demuxer); 
 * ...
 * asio::ipv4::address multicast_address("225.0.0.1");
 * asio::ipv4::multicast::add_membership option(multicast_address);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Setting the option to join a multicast group on the specified local
 * interface.
 * @code
 * asio::datagram_socket socket(demuxer); 
 * asio::ipv4::address multicast_address("225.0.0.1");
 * asio::ipv4::address local_interface("1.2.3.4");
 * asio::ipv4::multicast::add_membership option(
 *     multicast_address, local_interface);
 * socket.set_option(option);
 * ...
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, IPv4_MReq_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined add_membership;
#else
typedef asio::ipv4::detail::socket_option::multicast_request<
  IPPROTO_IP, IP_ADD_MEMBERSHIP> add_membership;
#endif

/// Socket option to leave a multicast group on a specified interface.
/**
 * Implements the IPPROTO_IP/IP_DROP_MEMBERSHIP socket option.
 *
 * @par Examples:
 * Setting the option to leave a multicast group:
 * @code
 * asio::datagram_socket socket(demuxer); 
 * ...
 * asio::ipv4::address multicast_address("225.0.0.1");
 * asio::ipv4::multicast::drop_membership option(multicast_address);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Setting the option to leave a multicast group on the specified local
 * interface.
 * @code
 * asio::datagram_socket socket(demuxer); 
 * asio::ipv4::address multicast_address("225.0.0.1");
 * asio::ipv4::address local_interface("1.2.3.4");
 * asio::ipv4::multicast::drop_membership option(
 *     multicast_address, local_interface);
 * socket.set_option(option);
 * ...
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, IPv4_MReq_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined drop_membership;
#else
typedef asio::ipv4::detail::socket_option::multicast_request<
  IPPROTO_IP, IP_DROP_MEMBERSHIP> drop_membership;
#endif

/// Socket option for local interface to use for outgoing multicast packets.
/**
 * Implements the IPPROTO_IP/IP_MULTICAST_IF socket option.
 *
 * @par Examples:
 * Setting the option:
 * @code
 * asio::datagram_socket socket(demuxer); 
 * ...
 * asio::ipv4::address local_interface("1.2.3.4");
 * asio::ipv4::multicast::outbound_interface option(local_interface);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Getting the current option value:
 * @code
 * asio::datagram_socket socket(demuxer); 
 * ...
 * asio::ipv4::multicast::outbound_interface option;
 * socket.get_option(option);
 * asio::ipv4::address local_interface = option.get();
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, IPv4_Address_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined outbound_interface;
#else
typedef asio::ipv4::detail::socket_option::address<
  IPPROTO_IP, IP_MULTICAST_IF> outbound_interface;
#endif

/// Socket option for time-to-live associated with outgoing multicast packets.
/**
 * Implements the IPPROTO_IP/IP_MULTICAST_TTL socket option.
 *
 * @par Examples:
 * Setting the option:
 * @code
 * asio::datagram_socket socket(demuxer); 
 * ...
 * asio::ipv4::multicast::time_to_live option(4);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Getting the current option value:
 * @code
 * asio::datagram_socket socket(demuxer); 
 * ...
 * asio::ipv4::multicast::time_to_live option;
 * socket.get_option(option);
 * int ttl = option.get();
 * @endcode
 *
 * @par Concepts:
 * Socket_Option, Integer_Socket_Option.
 */
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined time_to_live;
#else
typedef asio::detail::socket_option::integer<
  IPPROTO_IP, IP_MULTICAST_TTL> time_to_live;
#endif

/// Socket option determining whether outgoing multicast packets will be
/// received on the same socket if it is a member of the multicast group.
/**
 * Implements the IPPROTO_IP/IP_MULTICAST_LOOP socket option.
 *
 * @par Examples:
 * Setting the option:
 * @code
 * asio::datagram_socket socket(demuxer); 
 * ...
 * asio::ipv4::multicast::enable_loopback option(true);
 * socket.set_option(option);
 * @endcode
 *
 * @par
 * Getting the current option value:
 * @code
 * asio::datagram_socket socket(demuxer); 
 * ...
 * asio::ipv4::multicast::enable_loopback option;
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
  IPPROTO_IP, IP_MULTICAST_LOOP> enable_loopback;
#endif

} // namespace multicast
} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_MULTICAST_HPP
