//
// Error_Source.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Error source concept.
/**
 * @par Implemented By:
 * asio::basic_deadline_timer @n
 * asio::basic_datagram_socket @n
 * asio::basic_locking_dispatcher @n
 * asio::basic_socket_acceptor @n
 * asio::basic_stream_socket @n
 * asio::buffered_read_stream @n
 * asio::buffered_write_stream @n
 * asio::buffered_stream @n
 * asio::ipv4::basic_host_resolver @n
 * asio::ssl::stream
 */
class Error_Source
{
public:
  /// The type used for reporting errors.
  typedef implementation_defined error_type;
};
