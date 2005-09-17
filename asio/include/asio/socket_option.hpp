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

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

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
  std::size_t size() const
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
  std::size_t size() const
  {
    return sizeof(value_);
  }

private:
  int value_;
};

/// Helper template for implementing linger options.
template <int Level, int Name>
class linger
{
public:
  /// Default constructor.
  linger()
  {
    value_.l_onoff = 0;
    value_.l_linger = 0;
  }

  /// Construct with specific option values.
  linger(bool value, unsigned short timeout)
  {
    value_.l_onoff = value ? 1 : 0;
    value_.l_linger = timeout;
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

  /// Set the value for whether linger is enabled.
  void enabled(bool value)
  {
    value_.l_onoff = value ? 1 : 0;
  }

  /// Get the value for whether linger is enabled.
  bool enabled() const
  {
    return value_.l_onoff != 0;
  }

  /// Set the value for the linger timeout.
  void timeout(unsigned short value)
  {
    value_.l_linger = value;
  }

  /// Get the value for the linger timeout.
  unsigned short timeout() const
  {
    return value_.l_linger;
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
  std::size_t size() const
  {
    return sizeof(value_);
  }

private:
  ::linger value_;
};

} // namespace socket_option
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_OPTION_HPP
