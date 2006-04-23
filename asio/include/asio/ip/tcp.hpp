//
// tcp.hpp
// ~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_TCP_HPP
#define ASIO_IP_TCP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_resolver.hpp"
#include "asio/basic_socket_acceptor.hpp"
#include "asio/basic_stream_socket.hpp"
#include "asio/resolver_service.hpp"
#include "asio/socket_acceptor_service.hpp"
#include "asio/stream_socket_service.hpp"
#include "asio/ip/basic_endpoint.hpp"
#include "asio/ip/basic_resolver_iterator.hpp"
#include "asio/ip/basic_resolver_query.hpp"
#include "asio/ipv4/tcp.hpp"
#include "asio/ipv6/tcp.hpp"
#include "asio/detail/socket_option.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ip {

/// Encapsulates the flags needed for TCP.
/**
 * The asio::ip::tcp class contains flags necessary for TCP sockets.
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
  /// The IPv4 protocol type.
  typedef asio::ipv4::tcp ipv4_protocol;

  /// The IPv6 protocol type.
  typedef asio::ipv6::tcp ipv6_protocol;

  /// The type of a TCP endpoint.
  typedef basic_endpoint<tcp> endpoint;

  /// The type of a resolver query.
  typedef basic_resolver_query<tcp> resolver_query;

  /// The type of a resolver iterator.
  typedef basic_resolver_iterator<tcp> resolver_iterator;

  /// Construct to represent the IPv4 TCP protocol.
  tcp(const ipv4_protocol&)
    : family_(PF_INET)
  {
  }

  /// Construct to represent the IPv4 TCP protocol.
  tcp(const ipv6_protocol&)
    : family_(PF_INET6)
  {
  }

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
    return family_;
  }

  /// The service type for TCP sockets.
  typedef stream_socket_service<tcp> socket_service;

  /// The TCP socket type.
  typedef basic_stream_socket<socket_service> socket;

  /// The service type for TCP acceptors.
  typedef socket_acceptor_service<tcp> acceptor_service;

  /// The TCP acceptor type.
  typedef basic_socket_acceptor<acceptor_service> acceptor;

  /// The service type for TCP resolvers.
  typedef resolver_service<tcp> resolver_service;

  /// The TCP acceptor type.
  typedef basic_resolver<resolver_service> resolver;

  /// Socket option for disabling the Nagle algorithm.
  /**
   * Implements the IPPROTO_TCP/TCP_NODELAY socket option.
   *
   * @par Examples:
   * Setting the option:
   * @code
   * asio::ipv6::tcp::socket socket(io_service); 
   * ...
   * asio::ipv6::tcp::no_delay option(true);
   * socket.set_option(option);
   * @endcode
   *
   * @par
   * Getting the current option value:
   * @code
   * asio::ipv6::tcp::socket socket(io_service); 
   * ...
   * asio::ipv6::tcp::no_delay option;
   * socket.get_option(option);
   * bool is_set = option.get();
   * @endcode
   *
   * @par Concepts:
   * Socket_Option, Boolean_Socket_Option.
   */
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined no_delay;
#else
  typedef asio::detail::socket_option::boolean<
    IPPROTO_TCP, TCP_NODELAY> no_delay;
#endif

private:
  int family_;
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_TCP_HPP
