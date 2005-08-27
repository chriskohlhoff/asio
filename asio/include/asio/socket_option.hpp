//
// socket_option.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SOCKET_OPTION_HPP
#define ASIO_SOCKET_OPTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/socket_types.hpp"

namespace asio {
namespace socket_option {

/// Helper template for implementing boolean-based options.
template <int Level, int Name>
class boolean
{
public:
  /// Default constructor.
  boolean()
    : value_(0)
  {
  }

  /// Construct with a specific option value.
  boolean(bool value)
    : value_(value ? 1 : 0)
  {
  }

  /// Get the level of the socket option.
  int level() const
  {
    return Level;
  }

  /// Get the name of the socket option.
  int name() const
  {
    return Name;
  }

  /// Set the value of the boolean.
  void set(bool value)
  {
    value_ = value ? 1 : 0;
  }

  /// Get the current value of the boolean.
  bool get() const
  {
    return value_;
  }

  /// Get the address of the boolean data.
  void* data()
  {
    return &value_;
  }

  /// Get the address of the boolean data.
  const void* data() const
  {
    return &value_;
  }

  /// Get the size of the boolean data.
  size_t size() const
  {
    return sizeof(value_);
  }

private:
  int value_;
};

/// Helper template for implementing integer options.
template <int Level, int Name>
class integer
{
public:
  /// Default constructor.
  integer()
    : value_(0)
  {
  }

  /// Construct with a specific option value.
  integer(int value)
    : value_(value)
  {
  }

  /// Get the level of the socket option.
  int level() const
  {
    return Level;
  }

  /// Get the name of the socket option.
  int name() const
  {
    return Name;
  }

  /// Set the value of the int option.
  void set(int value)
  {
    value_ = value;
  }

  /// Get the current value of the int option.
  int get() const
  {
    return value_;
  }

  /// Get the address of the int data.
  void* data()
  {
    return &value_;
  }

  /// Get the address of the int data.
  const void* data() const
  {
    return &value_;
  }

  /// Get the size of the int data.
  size_t size() const
  {
    return sizeof(value_);
  }

private:
  int value_;
};

} // namespace socket_option
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_OPTION_HPP
