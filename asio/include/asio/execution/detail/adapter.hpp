//
// execution/detail/adapter.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_DETAIL_ADAPTER_HPP
#define ASIO_EXECUTION_DETAIL_ADAPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/can_query.hpp"
#include "asio/execution/detail/query_member_traits.hpp"
#include "asio/execution/detail/query_static_member_traits.hpp"
#include "asio/execution/detail/require_member_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <template <typename> typename Derived, typename Executor>
class adapter
{
public:
  adapter(Executor ex)
    : executor_(std::move(ex))
  {
  }

  template <typename Property>
  constexpr auto require(const Property& p) const
    noexcept(require_member_traits<Executor, Property>::is_noexcept)
    -> Derived<typename require_member_traits<Executor, Property>::result_type>
  {
    return Derived<decltype(executor_.require(p))>(executor_.require(p));
  }

  template <typename Property>
  static constexpr auto query(const Property& p)
    noexcept(query_static_member_traits<Executor, Property>::is_noexcept)
    -> typename query_static_member_traits<Executor, Property>::result_type
  {
    return Executor::query(p);
  }

  template <typename Property>
  constexpr auto query(const Property& p) const
    noexcept(query_member_traits<Executor, Property>::is_noexcept)
    -> typename enable_if<
      !query_static_member_traits<Executor, Property>::is_valid,
      typename query_member_traits<Executor, Property>::result_type
    >::type
  {
    return executor_.query(p);
  }

  friend constexpr bool operator==(const Derived<Executor>& a,
      const Derived<Executor>& b) noexcept
  {
    return a.executor_ == b.executor_;
  }

  friend constexpr bool operator!=(const Derived<Executor>& a,
      const Derived<Executor>& b) noexcept
  {
    return a.executor_ != b.executor_;
  }

protected:
  Executor executor_;
};

} // namespace detail
} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_DETAIL_ADAPTER_HPP
