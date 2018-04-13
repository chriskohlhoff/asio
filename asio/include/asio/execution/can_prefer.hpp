//
// execution/can_prefer.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_CAN_PREFER_HPP
#define ASIO_EXECUTION_CAN_PREFER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/detail/property_pack.hpp"
#include "asio/execution/detail/void_type.hpp"
#include "asio/execution/prefer.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename Executor, typename Properties, typename = void>
struct can_prefer : false_type
{
};

template <typename Executor, typename... Properties>
struct can_prefer<Executor, property_pack<Properties...>,
  typename void_type<
    decltype(asio::execution::prefer(declval<Executor>(),
      declval<Properties>()...))
  >::type> : true_type
{
};

} // namespace detail

template <typename Executor, typename... Properties>
struct can_prefer :
  detail::can_prefer<Executor, detail::property_pack<Properties...> >
{
};

template <typename Executor, typename... Properties>
constexpr bool can_prefer_v = can_prefer<Executor, Properties...>::value;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_CAN_PREFER_HPP
