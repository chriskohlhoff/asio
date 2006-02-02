//
// Protocol.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Protocol concept.
/**
 * Defines the interface that must be implemented by an object passed as the
 * @c protocol parameter to:
 * @li asio::socket_acceptor::open
 * @li asio::stream_socket::open
 * @li asio::datagram_socket::open
 *
 * @par Implemented By:
 * asio::ipv4::tcp @n
 * asio::ipv4::udp
 */
class Protocol
{
public:
  /// Obtain an identifier for the type of the protocol.
  int type() const;

  /// Obtain an identifier for the protocol.
  int protocol() const;

  /// Obtain an identifier for the protocol family.
  int family() const;
};
