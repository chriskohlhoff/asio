//
// IO_Control_Command.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// IO_Control_Command concept.
/**
 * Defines the interface that must be implemented by an object passed as the
 * @c command parameter to:
 * @li asio::stream_socket::io_control
 * @li asio::datagram_socket::io_control
 *
 * @par Implemented By:
 * asio::socket_base::non_blocking_io @n
 * asio::socket_base::bytes_readable
 */
class IO_Control_Command
{
public:
  /// Get the name of the command.
  int name() const;

  /// Get the data associated with the command.
  implementation_defined data();

  /// Get the data associated with the command.
  implementation_defined data() const;
};
