//
// execution/detail/twoway_adapter.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_DETAIL_TWOWAY_ADAPTER_HPP
#define ASIO_EXECUTION_DETAIL_TWOWAY_ADAPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <future>
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename Executor>
struct twoway_adapter
{
public:
  twoway_adapter(Executor ex)
    : executor_(std::move(ex))
  {
  }

  template <typename Property>
  auto require(const Property& p) const
    -> twoway_adapter<typename decay<
      decltype(declval<typename conditional<true,
        Executor, Property>::type>().require(p))>::type>
  {
    return twoway_adapter<typename decay<
      decltype(declval<typename conditional<true,
        Executor, Property>::type>().require(p))>::type>(
          executor_.require(p));
  }

  template <typename Property>
  auto query(const Property& p) const
    noexcept(noexcept(declval<typename conditional<true,
          Executor, Property>::type>().query(p)))
    -> decltype(declval<typename conditional<true,
        Executor, Property>::type>().query(p))
  {
    return executor_.query(p);
  }

  template <typename Function>
  auto execute(Function&& f)
    -> decltype(declval<typename conditional<true,
        Executor, Function>::type>().execute(declval<Function>()))
  {
    return executor_.execute(std::forward<Function>(f));
  }

  template <typename Function>
  auto twoway_execute(Function f)
    -> std::future<typename conditional<true, decltype(f()),
      decltype(declval<typename conditional<true, Executor,
        Function>::type>().execute(declval<Function>()))>::type>
  {
    std::packaged_task<decltype(f())()> task(std::move(f));
    std::future<decltype(f())> fut = task.get_future();
    executor_.execute(std::move(task));
    return fut;
  }

  friend constexpr bool operator==(
      const twoway_adapter& a, const twoway_adapter& b) noexcept
  {
    return a.executor_ == b.executor_;
  }

  friend constexpr bool operator!=(
      const twoway_adapter& a, const twoway_adapter& b) noexcept
  {
    return a.executor_ != b.executor_;
  }

private:
  template <typename> friend class twoway_adapter;
  Executor executor_;
};

} // namespace detail
} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_DETAIL_TWOWAY_ADAPTER_HPP
