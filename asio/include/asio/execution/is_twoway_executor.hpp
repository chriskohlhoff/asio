//
// execution/is_twoway_executor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_IS_TWOWAY_EXECUTOR_HPP
#define ASIO_EXECUTION_IS_TWOWAY_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <future>
#include "asio/detail/type_traits.hpp"
#include "asio/execution/detail/void_type.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

struct twoway_function
{
  int operator()()
  {
    return 0;
  }
};

template <typename T, typename = void>
struct is_twoway_executor : false_type
{
};

template <typename T>
struct is_twoway_executor<T,
  typename void_type<
    typename enable_if<std::is_nothrow_copy_constructible<T>::value>::type,
    typename enable_if<std::is_nothrow_move_constructible<T>::value>::type,
    typename enable_if<
      noexcept(static_cast<bool>(declval<const T&>() == declval<const T&>()))
    >::type,
    typename enable_if<
      noexcept(static_cast<bool>(declval<const T&>() != declval<const T&>()))
    >::type,
    typename enable_if<
      is_same<std::future<int>, decltype(declval<const T&>().twoway_execute(
       declval<twoway_function>()))>::value
    >::type,
    typename enable_if<
      is_same<std::future<int>, decltype(declval<const T&>().twoway_execute(
        declval<twoway_function&>()))>::value
    >::type,
    typename enable_if<
      is_same<std::future<int>, decltype(declval<const T&>().twoway_execute(
        declval<const twoway_function&>()))>::value
    >::type,
    typename enable_if<
      is_same<std::future<int>, decltype(declval<const T&>().twoway_execute(
        declval<twoway_function&&>()))>::value
    >::type
  >::type> : true_type
{
};

} // namespace detail

template <typename Executor>
struct is_twoway_executor : detail::is_twoway_executor<Executor>
{
};

template <typename Executor>
constexpr bool is_twoway_executor_v = is_twoway_executor<Executor>::value;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_IS_TWOWAY_EXECUTOR_HPP
