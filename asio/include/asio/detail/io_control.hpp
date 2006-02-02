//
// io_control.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IO_CONTROL_HPP
#define ASIO_DETAIL_IO_CONTROL_HPP

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
namespace detail {
namespace io_control {

// Helper template for implementing boolean-based IO control commands.
template <int Name>
class boolean
{
public:
  // Default constructor.
  boolean()
    : value_(0)
  {
  }

  // Construct with a specific command value.
  boolean(bool value)
    : value_(value ? 1 : 0)
  {
  }

  // Get the name of the IO control command.
  int name() const
  {
    return Name;
  }

  // Set the value of the boolean.
  void set(bool value)
  {
    value_ = value ? 1 : 0;
  }

  // Get the current value of the boolean.
  bool get() const
  {
    return value_ != 0;
  }

  // Get the address of the command data.
  detail::ioctl_arg_type* data()
  {
    return &value_;
  }

  // Get the address of the command data.
  const detail::ioctl_arg_type* data() const
  {
    return &value_;
  }

private:
  detail::ioctl_arg_type value_;
};

// Helper template for implementing size-based IO control commands.
template <int Name>
class size
{
public:
  // Default constructor.
  size()
    : value_(0)
  {
  }

  // Construct with a specific command value.
  size(std::size_t value)
    : value_(value)
  {
  }

  // Get the name of the IO control command.
  int name() const
  {
    return Name;
  }

  // Set the value of the size.
  void set(std::size_t value)
  {
    value_ = static_cast<detail::ioctl_arg_type>(value);
  }

  // Get the current value of the size.
  std::size_t get() const
  {
    return static_cast<std::size_t>(value_);
  }

  // Get the address of the command data.
  detail::ioctl_arg_type* data()
  {
    return &value_;
  }

  // Get the address of the command data.
  const detail::ioctl_arg_type* data() const
  {
    return &value_;
  }

private:
  detail::ioctl_arg_type value_;
};

} // namespace io_control
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IO_CONTROL_HPP
