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

/// Helper template for implementing flag-based options.
template <int Level, int Name>
class flag
{
public:
  /// Default constructor.
  flag()
    : value_(0)
  {
  }

  /// Construct to be either enabled or disabled.
  flag(bool enabled)
    : value_(enabled ? 1 : 0)
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

  /// Set the value of the flag.
  void set(bool enabled)
  {
    value_ = enabled ? 1 : 0;
  }

  /// Get the current value of the flag.
  bool get() const
  {
    return value_;
  }

  /// Get the address of the flag data.
  void* data()
  {
    return &value_;
  }

  /// Get the address of the flag data.
  const void* data() const
  {
    return &value_;
  }

  /// Get the size of the flag data.
  size_t size() const
  {
    return sizeof(value_);
  }

private:
  /// The underlying value of the flag.
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
  /// The underlying value of the int option.
  int value_;
};

/// Permit sending of broadcast messages.
typedef flag<SOL_SOCKET, SO_BROADCAST> broadcast;

/// Prevent routing, use local interfaces only.
typedef flag<SOL_SOCKET, SO_DONTROUTE> dont_route;

/// Send keep-alives.
typedef flag<SOL_SOCKET, SO_KEEPALIVE> keep_alive;

/// The receive buffer size for a socket.
typedef integer<SOL_SOCKET, SO_SNDBUF> send_buffer_size;

/// Send low watermark.
typedef integer<SOL_SOCKET, SO_SNDLOWAT> send_low_watermark;

/// Send timeout.
typedef integer<SOL_SOCKET, SO_SNDTIMEO> send_timeout;

/// The send buffer size for a socket.
typedef integer<SOL_SOCKET, SO_RCVBUF> recv_buffer_size;

/// Receive low watermark.
typedef integer<SOL_SOCKET, SO_RCVLOWAT> recv_low_watermark;

/// Receive timeout.
typedef integer<SOL_SOCKET, SO_RCVTIMEO> recv_timeout;

/// Allow the socket to be bound to an address that is already in use.
typedef flag<SOL_SOCKET, SO_REUSEADDR> reuse_address;

} // namespace socket_option
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_OPTION_HPP
