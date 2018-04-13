//
// execution/prefer.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_PREFER_HPP
#define ASIO_EXECUTION_PREFER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <utility>
#include "asio/execution/detail/require_static_traits.hpp"
#include "asio/execution/detail/require_member_traits.hpp"
#include "asio/execution/detail/prefer_free_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {
namespace prefer_fn {

struct impl
{
  template <typename Executor, typename Property>
  constexpr auto operator()(Executor&& ex, Property&&) const
    -> typename enable_if<
      decay<Property>::type::is_preferable
        && require_static_traits<Executor, Property>::is_valid,
      typename require_static_traits<Executor, Property>::result_type
    >::type
  {
    return std::forward<Executor>(ex);
  }

  template <typename Executor, typename Property>
  constexpr auto operator()(Executor&& ex, Property&& p) const
    noexcept(require_member_traits<Executor, Property>::is_noexcept)
    -> typename enable_if<
      decay<Property>::type::is_preferable
        && !require_static_traits<Executor, Property>::is_valid
        && require_member_traits<Executor, Property>::is_valid,
      typename require_member_traits<Executor, Property>::result_type
    >::type
  {
    return std::forward<Executor>(ex).prefer(std::forward<Property>(p));
  }

  template <typename Executor, typename Property>
  constexpr auto operator()(Executor&& ex, Property&& p) const
    noexcept(prefer_free_traits<Executor, Property>::is_noexcept)
    -> typename enable_if<
      decay<Property>::type::is_preferable
        && !require_static_traits<Executor, Property>::is_valid
        && !require_member_traits<Executor, Property>::is_valid
        && prefer_free_traits<Executor, Property>::is_valid,
      typename prefer_free_traits<Executor, Property>::result_type
    >::type
  {
    return prefer(std::forward<Executor>(ex), std::forward<Property>(p));
  }

  template <typename Executor, typename Property>
  constexpr auto operator()(Executor&& ex, Property&& p) const
    -> typename enable_if<
      decay<Property>::type::is_preferable
        && !require_static_traits<Executor, Property>::is_valid
        && !require_member_traits<Executor, Property>::is_valid
        && !prefer_free_traits<Executor, Property>::is_valid,
      typename decay<Executor>::type
    >::type
  {
    return std::forward<Executor>(ex);
  }

  template <typename Executor, typename Property0,
      typename Property1, typename... PropertyN>
  constexpr auto operator()(Executor&& ex, Property0&& p0,
      Property1&& p1, PropertyN&&... pn) const
    noexcept(noexcept(declval<impl>()(
      declval<impl>()(std::forward<Executor>(ex), std::forward<Property0>(p0)),
        std::forward<Property1>(p1), std::forward<PropertyN>(pn)...)))
    -> decltype(declval<impl>()(
      declval<impl>()(std::forward<Executor>(ex), std::forward<Property0>(p0)),
        std::forward<Property1>(p1), std::forward<PropertyN>(pn)...))
  {
    return (*this)(
      (*this)(std::forward<Executor>(ex), std::forward<Property0>(p0)),
        std::forward<Property1>(p1), std::forward<PropertyN>(pn)...);
  }
};

template <typename T = impl>
constexpr T customization_point = T{};

} // namespace prefer_fn
} // namespace detail
namespace {

constexpr const auto& prefer = detail::prefer_fn::customization_point<>;

} // namespace
} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_PREFER_HPP
