//
// IP_MReq_Socket_Option.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// IP_MReq_Socket_Option concept for performing multicast requests.
/**
 * @par Implemented By:
 * asio::ip::multicast::join_group @n
 * asio::ip::multicast::leave_group
 */
class IP_MReq_Socket_Option
  : public Socket_Option
{
public:
  /// Default constructor initialises both the multicast address and the
  /// local interface to the "any" address.
  IP_MReq_Socket_Option();

  /// Construct with multicast address only.
  IP_MReq_Socket_Option(const asio::ip::address& multicast_address);

  /// Construct with IP version 4 multicast address and address of local
  /// interface to use.
  IP_MReq_Socket_Option(const asio::ip::address_v4& multicast_address,
      const asio::ip::address_v4& local_interface);

  /// Construct with IP version 6 multicast address and network interface index.
  IP_MReq_Socket_Option(const asio::ip::address_v6& multicast_address,
      unsigned long local_interface);
};
