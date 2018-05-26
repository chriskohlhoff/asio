//
// execution/twoway.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_TWOWAY_HPP
#define ASIO_EXECUTION_TWOWAY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <future>
#include "asio/detail/type_traits.hpp"
#include "asio/execution/blocking_adaptation.hpp"
#include "asio/execution/detail/adapter.hpp"
#include "asio/execution/is_oneway_executor.hpp"
#include "asio/execution/is_twoway_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct twoway_t
{
  /// The @c twoway property may be used with asio::execution::require.
  static constexpr bool is_requirable = true;

  /// The @c twoway property cannot be used with asio::execution::prefer.
  static constexpr bool is_preferable = false;

  /// The @c twoway property is polymorphically stored as @c bool.
  using polymorphic_query_result_type = bool;

  /// Statically determines whether an Executor type provides the @c twoway
  /// property.
  template <typename Executor>
  static constexpr bool static_query_v = is_twoway_executor<Executor>::value;

  /// Obtain the value associated with the twoway property. Always @c true.
  static constexpr bool value()
  {
    return true;
  }

private:
  template <typename Executor>
  class adapter : public detail::adapter<adapter, Executor>
  {
  public:
    using detail::adapter<adapter, Executor>::adapter;

    template <typename Function>
    auto execute(Function&& f) const
      -> decltype(declval<typename conditional<true,
          Executor, Function>::type>().execute(declval<Function>()))
    {
      return this->executor_.execute(std::forward<Function>(f));
    }

    template <typename Function>
    auto twoway_execute(Function f) const
      -> std::future<typename conditional<true, decltype(f()),
        decltype(declval<typename conditional<true, Executor,
          Function>::type>().execute(declval<Function>()))>::type>
    {
      std::packaged_task<decltype(f())()> task(std::move(f));
      std::future<decltype(f())> fut = task.get_future();
      this->executor_.execute(std::move(task));
      return fut;
    }
  };

public:
  /// Default adapter adapts oneway as twoway.
  template <typename Executor, typename =
      typename enable_if<
        is_oneway_executor<Executor>::value
          && !is_twoway_executor<Executor>::value
          && blocking_adaptation_t::static_query_v<Executor>
            == blocking_adaptation.allowed
      >::type>
  friend adapter<Executor> require(Executor ex, twoway_t)
  {
    return adapter<Executor>(std::move(ex));
  }
};

constexpr twoway_t twoway;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_TWOWAY_HPP
