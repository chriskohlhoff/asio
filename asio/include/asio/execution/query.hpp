//
// execution/query.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_QUERY_HPP
#define ASIO_EXECUTION_QUERY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <utility>
#include "asio/execution/detail/query_static_traits.hpp"
#include "asio/execution/detail/query_member_traits.hpp"
#include "asio/execution/detail/query_free_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {
namespace query_fn {

struct impl
{
  template <typename Executor, typename Property>
  constexpr auto operator()(Executor&& ex, Property&&) const
    -> typename enable_if<
      query_static_traits<Executor, Property>::is_valid,
      typename query_static_traits<Executor, Property>::result_type
    >::type
  {
    return std::forward<Executor>(ex);
  }

  template <typename Executor, typename Property>
  constexpr auto operator()(Executor&& ex, Property&& p) const
    noexcept(query_member_traits<Executor, Property>::is_noexcept)
    -> typename enable_if<
      !query_static_traits<Executor, Property>::is_valid
        && query_member_traits<Executor, Property>::is_valid,
      typename query_member_traits<Executor, Property>::result_type
    >::type
  {
    return std::forward<Executor>(ex).query(std::forward<Property>(p));
  }

  template <typename Executor, typename Property>
  constexpr auto operator()(Executor&& ex, Property&& p) const
    noexcept(query_free_traits<Executor, Property>::is_noexcept)
    -> typename enable_if<
      !query_static_traits<Executor, Property>::is_valid
        && !query_member_traits<Executor, Property>::is_valid
        && query_free_traits<Executor, Property>::is_valid,
      typename query_free_traits<Executor, Property>::result_type
    >::type
  {
    return query(std::forward<Executor>(ex), std::forward<Property>(p));
  }
};

template <typename T = impl>
constexpr T customization_point = T{};

} // namespace query_fn
} // namespace detail
namespace {

constexpr const auto& query = detail::query_fn::customization_point<>;

} // namespace
} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_QUERY_HPP
