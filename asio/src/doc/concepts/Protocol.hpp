//
// Protocol.hpp
// ~~~~~~~~~~~~
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

/// Protocol concept.
/**
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
