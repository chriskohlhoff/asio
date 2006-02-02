//
// IPv4_MReq_Socket_Option.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// IPv4_MReq_Socket_Option concept for performing multicast requests.
/**
 * @par Implemented By:
 * asio::ipv4::multicast::add_membership @n
 * asio::ipv4::multicast::drop_membership
 */
class IPv4_MReq_Socket_Option
  : public Socket_Option
{
public:
  /// Default constructor initialises both the multicast address and the
  /// local interface to asio::ipv4::address::any().
  IPv4_MReq_Socket_Option();

  /// Construct with multicast address only.
  IPv4_MReq_Socket_Option(const asio::ipv4::address& multicast_address);

  /// Construct with multicast address and address of local interface to use.
  IPv4_MReq_Socket_Option(const asio::ipv4::address& multicast_address,
      const asio::ipv4::address& local_interface);
};
