//
// udp.hpp
// ~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_IPV4_UDP_HPP
#define ASIO_IPV4_UDP_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv4 {

/// Encapsulates the flags needed for UDP.
/**
 * The asio::ipv4::udp class contains the flags necessary to use UDP sockets.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe.
 *
 * @par Concepts:
 * Protocol.
 */
class udp
{
public:
  /// Obtain an identifier for the type of the protocol.
  int type() const
  {
    return SOCK_DGRAM;
  }

  /// Obtain an identifier for the protocol.
  int protocol() const
  {
    return IPPROTO_UDP;
  }

  /// Obtain an identifier for the protocol family.
  int family() const
  {
    return PF_INET;
  }
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_UDP_HPP
