//
// tcp.hpp
// ~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_TCP_HPP
#define ASIO_IPV4_TCP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <memory>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_socket_acceptor.hpp"
#include "asio/basic_stream_socket.hpp"
#include "asio/socket_acceptor_service.hpp"
#include "asio/stream_socket_service.hpp"
#include "asio/ipv4/basic_endpoint.hpp"
#include "asio/detail/socket_option.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv4 {

/// Encapsulates the flags needed for TCP.
/**
 * The asio::ipv4::tcp class contains flags necessary for TCP sockets.
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
  /// The type of a TCP endpoint.
  typedef basic_endpoint<tcp> endpoint;

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

  /// Template typedefs for acceptor and socket types.
  template <typename Allocator>
  struct types
  {
    /// The service type for IPv4 TCP sockets.
    typedef stream_socket_service<tcp, Allocator> socket_service;

    /// The IPv4 TCP socket type.
    typedef basic_stream_socket<socket_service> socket;

    /// The service type for IPv4 TCP acceptors.
    typedef socket_acceptor_service<tcp, Allocator> acceptor_service;

    /// The IPv4 TCP acceptor type.
    typedef basic_socket_acceptor<acceptor_service> acceptor;
  };

  /// The IPv4 TCP socket type.
  typedef types<std::allocator<void> >::socket socket;

  /// The IPv4 TCP acceptor type.
  typedef types<std::allocator<void> >::acceptor acceptor;

  /// Socket option for disabling the Nagle algorithm.
  /**
   * Implements the IPPROTO_TCP/TCP_NODELAY socket option.
   *
   * @par Examples:
   * Setting the option:
   * @code
   * asio::ipv4::tcp::socket socket(io_service); 
   * ...
   * asio::ipv4::tcp::no_delay option(true);
   * socket.set_option(option);
   * @endcode
   *
   * @par
   * Getting the current option value:
   * @code
   * asio::ipv4::tcp::socket socket(io_service); 
   * ...
   * asio::ipv4::tcp::no_delay option;
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
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_TCP_HPP
