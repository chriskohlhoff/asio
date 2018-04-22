//
// execution/executor_future.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_EXECUTOR_FUTURE_HPP
#define ASIO_EXECUTION_EXECUTOR_FUTURE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/require.hpp"
#include "asio/execution/twoway.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

template <typename Executor, typename T>
struct executor_future
{
  using type = decltype(
      execution::require(
        declval<const Executor&>(),
        execution::twoway
      ).twoway_execute(declval<T(*)()>()));
};

template <typename Executor, typename T>
using executor_future_t = typename executor_future<Executor, T>::type;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_EXECUTOR_FUTURE_HPP
