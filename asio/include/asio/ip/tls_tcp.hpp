//
// ip/tls_tcp.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_TLS_TCP_HPP
#define ASIO_IP_TLS_TCP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/basic_socket_acceptor.hpp"
#include "asio/basic_stream_socket.hpp"
#include "asio/detail/apple_nw_ptr.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/ip/basic_endpoint.hpp"
#include "asio/ip/basic_resolver.hpp"
#include "asio/ip/basic_resolver_iterator.hpp"
#include "asio/ip/basic_resolver_query.hpp"
#include <Network/Network.h>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

/// Encapsulates the flags needed for TCP.
/**
 * The asio::ip::tls_tcp class contains flags necessary for TLS/TCP
 * sockets.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe.
 *
 * @par Concepts:
 * Protocol, InternetProtocol.
 */
class tls_tcp
{
public:
  /// The type of a TLS/TCP endpoint.
  typedef basic_endpoint<tls_tcp> endpoint;

  /// Construct to represent the IPv4 TLS/TCP protocol.
  static tls_tcp v4() ASIO_NOEXCEPT
  {
    return tls_tcp(ASIO_OS_DEF(AF_INET));
  }

  /// Construct to represent the IPv6 TLS/TCP protocol.
  static tls_tcp v6() ASIO_NOEXCEPT
  {
    return tls_tcp(ASIO_OS_DEF(AF_INET6));
  }

  /// Construct to represent an unspecified TLS/TCP protocol.
  static tls_tcp any() ASIO_NOEXCEPT
  {
    return tls_tcp(ASIO_OS_DEF(AF_UNSPEC));
  }

  /// Obtain an identifier for the type of the protocol.
  int type() const ASIO_NOEXCEPT
  {
    return ASIO_OS_DEF(SOCK_STREAM);
  }

  /// Obtain an identifier for the protocol.
  int protocol() const ASIO_NOEXCEPT
  {
    return ASIO_OS_DEF(IPPROTO_TCP);
  }

  /// Obtain an identifier for the protocol family.
  int family() const ASIO_NOEXCEPT
  {
    return family_;
  }

  // Obtain parameters to be used when creating a new connection or listener.
  ASIO_DECL asio::detail::apple_nw_ptr<nw_parameters_t>
  apple_nw_create_parameters() const;

  // Obtain the override value for the maximum receive size.
  std::size_t apple_nw_max_receive_size() const ASIO_NOEXCEPT
  {
    return 0;
  }

  /// The TCP socket type.
  typedef basic_stream_socket<tls_tcp> socket;

  /// The TCP acceptor type.
  typedef basic_socket_acceptor<tls_tcp> acceptor;

  /// The TCP resolver type.
  typedef basic_resolver<tls_tcp> resolver;

  /// Compare two protocols for equality.
  friend bool operator==(const tls_tcp& p1, const tls_tcp& p2)
  {
    return p1.family_ == p2.family_;
  }

  /// Compare two protocols for inequality.
  friend bool operator!=(const tls_tcp& p1, const tls_tcp& p2)
  {
    return p1.family_ != p2.family_;
  }

private:
  // Construct with a specific family.
  explicit tls_tcp(int protocol_family) ASIO_NOEXCEPT
    : family_(protocol_family)
  {
  }

  int family_;
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ip/impl/tls_tcp.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_IP_TLS_TCP_HPP
