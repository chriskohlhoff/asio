//
// detail/channel_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

class channel_op_base
  : public operation
{
public:
  // Mark the operation as successful.
  void on_cancel()
  {
    index_ = cancelled;
  }

  // Mark the operation as broken.
  void on_close()
  {
    index_ = closed;
  }

  // Determine whether the operation contains a value.
  bool has_value() const
  {
    return index_ >= 0;
  }

protected:
  channel_op_base(func_type func)
    : operation(func),
      has_value_(false),
      result_(success)
  {
  }

  // What is currently held by the operation. Values are non-negative.
  int index_;
  enum { no_value = -1 };
  enum { cancelled = -2 };
  enum { closed = -3 };
};

template <typename T>
class channel_op
  : public channel_op_base
{
public:
  // Retrieve the value.
  T get_value()
  {
    if (!has_value_)
      return T();

    return ASIO_MOVE_CAST(T)(*static_cast<T*>(
          static_cast<void*>(&value_)));
  }

  // Construct the value.
  template <typename T0>
  void set_value(ASIO_MOVE_ARG(T0) value)
  {
    new (&value_) T(ASIO_MOVE_CAST(T0)(value));
    has_value_ = true;
  }

protected:
  channel_op(func_type func)
    : channel_op_base(func)
  {
  }

  template <typename T0>
  channel_op(ASIO_MOVE_ARG(T0) value, func_type func)
    : channel_op_base(func)
  {
    this->set_value(ASIO_MOVE_CAST(T0)(value));
  }

  ~channel_op()
  {
    if (has_value_)
      get_value().~T();
  }

private:
  // The value to be passed through the channel.
  typename aligned_storage<sizeof(T)>::type value_;
};

template <>
class channel_op<void>
  : public channel_op_base
{
public:
  // Retrieve the value.
  int get_value()
  {
    return 0;
  }

  // Construct the value.
  void set_value(int)
  {
  }

protected:
  channel_op(int, func_type func)
    : channel_op_base(func)
  {
  }

  channel_op(func_type func)
    : channel_op_base(func)
  {
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_CHANNEL_OP_HPP
