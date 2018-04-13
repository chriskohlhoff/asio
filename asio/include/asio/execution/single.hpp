//
// execution/single.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_SINGLE_HPP
#define ASIO_EXECUTION_SINGLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/is_oneway_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct single_t
{
  /// The @c single property may be used with asio::execution::require.
  static constexpr bool is_requirable = true;

  /// The @c single property cannot be used with asio::execution::prefer.
  static constexpr bool is_preferable = false;

  /// The @c single property is polymorphically stored as @c bool.
  using polymorphic_query_result_type = bool;

  /// Statically determines whether an Executor type provides the @c single
  /// property.
  template <typename Executor>
  static constexpr bool static_query_v = is_oneway_executor<Executor>::value;

  /// Obtain the value associated with the single property. Always @c true.
  static constexpr bool value()
  {
    return true;
  }
};

constexpr single_t single;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_SINGLE_HPP
