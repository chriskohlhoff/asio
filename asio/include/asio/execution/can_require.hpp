//
// execution/can_require.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_CAN_REQUIRE_HPP
#define ASIO_EXECUTION_CAN_REQUIRE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/detail/property_pack.hpp"
#include "asio/execution/detail/void_type.hpp"
#include "asio/execution/require.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename Executor, typename Properties, typename = void>
struct can_require : false_type
{
};

template <typename Executor, typename... Properties>
struct can_require<Executor, property_pack<Properties...>,
  typename void_type<
    decltype(asio::execution::require(declval<Executor>(),
      declval<Properties>()...))
  >::type> : true_type
{
};

} // namespace detail

template <typename Executor, typename... Properties>
struct can_require :
  detail::can_require<Executor, detail::property_pack<Properties...> >
{
};

template <typename Executor, typename... Properties>
constexpr bool can_require_v = can_require<Executor, Properties...>::value;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_CAN_REQUIRE_HPP
