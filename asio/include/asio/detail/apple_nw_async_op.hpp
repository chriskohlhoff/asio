//
// detail/apple_nw_async_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_APPLE_NW_ASYNC_OP_HPP
#define ASIO_DETAIL_APPLE_NW_ASYNC_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/error.hpp"
#include "asio/detail/operation.hpp"
#include <Network/Network.h>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
class apple_nw_async_op
  : public operation
{
public:
  void set(const asio::error_code& ec, T result)
  {
    ec_ = ec;
    result_ = ASIO_MOVE_CAST(T)(result);
  }

  void set(nw_error_t error, T result)
  {
    set(asio::error_code(error ? nw_error_get_error_code(error) : 0,
          asio::system_category()), ASIO_MOVE_CAST(T)(result));
  }

protected:
  apple_nw_async_op(func_type complete_func)
    : operation(complete_func),
      result_()
  {
  }

  // The error code to be passed to the completion handler.
  asio::error_code ec_;

  // The result of the operation, to be passed to the completion handler.
  T result_;
};

template <>
class apple_nw_async_op<void>
  : public operation
{
public:
  void set(const asio::error_code& ec)
  {
    ec_ = ec;
  }

  void set(nw_error_t error)
  {
    set(asio::error_code(
          error ? nw_error_get_error_code(error) : 0,
          asio::system_category()));
  }

protected:
  apple_nw_async_op(func_type complete_func)
    : operation(complete_func)
  {
  }

  // The error code to be passed to the completion handler.
  asio::error_code ec_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_DETAIL_APPLE_NW_ASYNC_OP_HPP
