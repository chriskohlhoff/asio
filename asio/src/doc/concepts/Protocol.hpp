//
// Protocol.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Protocol concept.
/**
 * @par Implemented By:
 * asio::ip::tcp @n
 * asio::ip::udp
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
