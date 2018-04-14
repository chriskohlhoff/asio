//
// execution/detail/query_static_traits.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_DETAIL_QUERY_STATIC_TRAITS_HPP
#define ASIO_EXECUTION_DETAIL_QUERY_STATIC_TRAITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/detail/void_type.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename Executor, typename Property, typename = void>
struct query_static_traits
{
  static constexpr bool is_valid = false;
  static constexpr bool is_noexcept = false;
};

template <typename Executor, typename Property>
auto foo() -> decltype(
  decay<Property>::type::template static_query_v<
    typename decay<Executor>::type>);

template <typename Executor, typename Property>
struct query_static_traits<Executor, Property,
  typename void_type<
    decltype(
      decay<Property>::type::template static_query_v<
        typename decay<Executor>::type>)
  >::type>
{
  static constexpr bool is_valid = true;

  using result_type = decltype(
    decay<Property>::type::template static_query_v<
      typename decay<Executor>::type>);

  static constexpr bool is_noexcept = true;
};

} // namespace detail
} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_DETAIL_QUERY_STATIC_TRAITS_HPP
