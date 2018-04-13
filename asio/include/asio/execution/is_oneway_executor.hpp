//
// execution/is_oneway_executor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_IS_ONEWAY_EXECUTOR_HPP
#define ASIO_EXECUTION_IS_ONEWAY_EXECUTOR_HPP

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

struct oneway_function
{
  void operator()()
  {
  }
};

template <typename T, typename = void>
struct is_oneway_executor : false_type
{
};

template <typename T>
struct is_oneway_executor<T,
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
      is_same<void, decltype(declval<const T&>().execute(
       declval<oneway_function>()))>::value
    >::type,
    typename enable_if<
      is_same<void, decltype(declval<const T&>().execute(
        declval<oneway_function&>()))>::value
    >::type,
    typename enable_if<
      is_same<void, decltype(declval<const T&>().execute(
        declval<const oneway_function&>()))>::value
    >::type,
    typename enable_if<
      is_same<void, decltype(declval<const T&>().execute(
        declval<oneway_function&&>()))>::value
    >::type
  >::type> : true_type
{
};

} // namespace detail

template <typename Executor>
struct is_oneway_executor : detail::is_oneway_executor<Executor>
{
};

template <typename Executor>
constexpr bool is_oneway_executor_v = is_oneway_executor<Executor>::value;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_IS_ONEWAY_EXECUTOR_HPP
