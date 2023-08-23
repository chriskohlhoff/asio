//
// execution/detail/as_receiver.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_DETAIL_AS_RECEIVER_HPP
#define ASIO_EXECUTION_DETAIL_AS_RECEIVER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/traits/set_done_member.hpp"
#include "asio/traits/set_error_member.hpp"
#include "asio/traits/set_value_member.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename Function, typename>
struct as_receiver
{
  Function f_;

  template <typename F>
  explicit as_receiver(F&& f, int)
    : f_(static_cast<F&&>(f))
  {
  }

#if defined(ASIO_MSVC)
  as_receiver(as_receiver&& other)
    : f_(static_cast<Function&&>(other.f_))
  {
  }
#endif // defined(ASIO_MSVC)

  void set_value()
    noexcept(noexcept(declval<Function&>()()))
  {
    f_();
  }

  template <typename E>
  void set_error(E) noexcept
  {
    std::terminate();
  }

  void set_done() noexcept
  {
  }
};

template <typename T>
struct is_as_receiver : false_type
{
};

template <typename Function, typename T>
struct is_as_receiver<as_receiver<Function, T>> : true_type
{
};

} // namespace detail
} // namespace execution
namespace traits {

#if !defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

template <typename Function, typename T>
struct set_value_member<
    asio::execution::detail::as_receiver<Function, T>, void()>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = noexcept(declval<Function&>()());
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)

template <typename Function, typename T, typename E>
struct set_error_member<
    asio::execution::detail::as_receiver<Function, T>, E>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_SET_DONE_MEMBER_TRAIT)

template <typename Function, typename T>
struct set_done_member<
    asio::execution::detail::as_receiver<Function, T>>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_SET_DONE_MEMBER_TRAIT)

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_DETAIL_AS_RECEIVER_HPP
