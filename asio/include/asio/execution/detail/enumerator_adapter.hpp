//
// execution/detail/enumerator_adapter.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_DETAIL_ENUMERATOR_ADAPTER_HPP
#define ASIO_EXECUTION_DETAIL_ENUMERATOR_ADAPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/detail/adapter.hpp"
#include "asio/execution/require.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <template <typename> typename Derived, typename Executor,
    typename Enumeration, typename Enumerator>
class enumerator_adapter : public adapter<Derived, Executor>
{
public:
  using adapter<Derived, Executor>::adapter;
  using adapter<Derived, Executor>::require;
  using adapter<Derived, Executor>::query;

  template <int N>
  constexpr auto require(
      const typename Enumeration::template enumerator<N>& p) const
    -> decltype(execution::require(declval<Executor>(), p))
  {
    return execution::require(this->executor_, p);
  }

  static constexpr Enumeration query(const Enumeration&) noexcept
  {
    return Enumerator{};
  }

  template <int N>
  static constexpr Enumeration query(
      const typename Enumeration::template enumerator<N>&) noexcept
  {
    return Enumerator{};
  }
};

} // namespace detail
} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_DETAIL_ENUMERATOR_ADAPTER_HPP
