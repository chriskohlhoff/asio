//
// execution/context.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_CONTEXT_PROPERTY_HPP
#define ASIO_EXECUTION_CONTEXT_PROPERTY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct context_t
{
  /// The @c context property may be used with asio::execution::require.
  static constexpr bool is_requirable = true;

  /// The @c context property cannot be used with asio::execution::prefer.
  static constexpr bool is_preferable = false;

  /// The @c context property is polymorphically stored as @c bool.
  // using polymorphic_query_result_type = ...;

  /// Statically determines whether an Executor type provides the @c context
  /// property.
  template <typename Executor, typename Property = context_t,
    typename T = typename enable_if<
      (static_cast<void>(Executor::query(*static_cast<Property*>(0))), true),
      decltype(Executor::query(*static_cast<Property*>(0)))
    >::type>
  static constexpr T static_query_v = Executor::query(Property());
};

constexpr context_t context;

template <typename Executor, typename Property, typename T>
constexpr T context_t::static_query_v;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_CONTEXT_PROPERTY_HPP
