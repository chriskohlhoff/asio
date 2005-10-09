//
// Mutable_Buffers.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Mutable_Buffers concept.
/**
 * @par Implemented By:
 * asio::mutable_buffers @n
 * std::deque<asio::mutable_buffer> @n
 * std::list<asio::mutable_buffer> @n
 * std::vector<asio::mutable_buffer> @n
 * boost::array<asio::mutable_buffer, N>
 */
class Mutable_Buffers
{
public:
  /// The type for each element in the list of buffers. The type must be
  /// asio::mutable_buffer or be convertible to asio::mutable_buffer.
  typedef implementation_defined value_type;

  /// A forward iterator type that may be used to read or modify elements.
  typedef implementation_defined iterator;

  /// A forward iterator type that may be used to read elements.
  typedef implementation_defined const_iterator;

  /// Get an iterator to the first element.
  iterator begin();

  /// Get an iterator to the first element.
  const_iterator begin() const;

  /// Get an iterator for one past the last element.
  iterator end();

  /// Get an iterator for one past the last element.
  const_iterator end() const;
};
