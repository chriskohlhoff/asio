//
// Boolean_IO_Control_Command.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Boolean_IO_Control_Command concept.
/**
 * @par Implemented By:
 * asio::socket_base::non_blocking_io
 */
class Boolean_IO_Control_Command
  : public IO_Control_Command
{
public:
  /// Default constructor initialises boolean value to false.
  Boolean_IO_Control_Command();

  /// Construct with a specific command value.
  Boolean_IO_Control_Command(bool value);

  /// Set the value of the boolean.
  void set(bool value);

  /// Get the current value of the boolean.
  bool get() const;
};
