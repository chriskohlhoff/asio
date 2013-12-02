//
// detail/channel_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CHANNEL_OP_HPP
#define ASIO_DETAIL_CHANNEL_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/operation.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
class channel_op
  : public operation
{
public:
  // The error code to be passed to the completion handler.
  asio::error_code ec_;

  // Retrieve the value.
  T& get_value()
  {
    return *static_cast<T*>(static_cast<void*>(&value_));
  }

  // Default-construct the value.
  void set_default_value()
  {
    new (&value_) T;
    initialised_ = true;
  }

  // Construct the value.
  template <typename T0>
  void set_value(ASIO_MOVE_ARG(T0) value)
  {
    new (&value_) T(ASIO_MOVE_CAST(T0)(value));
    initialised_ = true;
  }

  // Determine whether the operation contains a value.
  bool has_value() const
  {
    return initialised_;
  }

protected:
  channel_op(func_type func)
    : operation(func),
      initialised_(false)
  {
  }

  template <typename T0>
  channel_op(ASIO_MOVE_ARG(T0) value, func_type func)
    : operation(func)
  {
    this->set_value(ASIO_MOVE_CAST(T0)(value));
  }

  ~channel_op()
  {
    if (initialised_)
      get_value().~T();
  }

private:
  // The value to be passed through the channel.
  typename aligned_storage<sizeof(T)>::type value_;

  // Whether the value has been initialised.
  bool initialised_;
};

template <>
class channel_op<void>
  : public operation
{
public:
  // The error code to be passed to the completion handler.
  asio::error_code ec_;

  // Retrieve the value.
  void get_value()
  {
  }

  // Default-construct the value.
  void set_default_value()
  {
    initialised_ = true;
  }

  // Construct the value.
  void set_value()
  {
    initialised_ = true;
  }

  // Determine whether the operation contains a value.
  bool has_value() const
  {
    return initialised_;
  }

protected:
  channel_op(func_type func)
    : operation(func),
      initialised_(false)
  {
  }

private:
  // Whether the value has been initialised.
  bool initialised_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_CHANNEL_OP_HPP
