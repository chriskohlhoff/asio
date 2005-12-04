//
// Size_IO_Control_Command.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Size_IO_Control_Command concept.
/**
 * @par Implemented By:
 * asio::socket_base::bytes_readable
 */
class Size_IO_Control_Command
  : public IO_Control_Command
{
public:
  /// Default constructor initialises size value to 0.
  Size_IO_Control_Command();

  /// Construct with a specific command value.
  Size_IO_Control_Command(std::size_t value);

  /// Set the size value.
  void set(std::size_t value);

  /// Get the current size value.
  std::size_t get() const;
};
