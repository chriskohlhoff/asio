//
// execution/set_value.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_SET_VALUE_HPP
#define ASIO_EXECUTION_SET_VALUE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_DEPRECATED)

#include "asio/detail/type_traits.hpp"
#include "asio/traits/set_value_member.hpp"
#include "asio/traits/set_value_free.hpp"

#include "asio/detail/push_options.hpp"

#if defined(GENERATING_DOCUMENTATION)

namespace asio {
namespace execution {

/// A customisation point that delivers a value to a receiver.
/**
 * The name <tt>execution::set_value</tt> denotes a customisation point object.
 * The expression <tt>execution::set_value(R, Vs...)</tt> for some
 * subexpressions <tt>R</tt> and <tt>Vs...</tt> is expression-equivalent to:
 *
 * @li <tt>R.set_value(Vs...)</tt>, if that expression is valid. If the
 *   function selected does not send the value(s) <tt>Vs...</tt> to the receiver
 *   <tt>R</tt>'s value channel, the program is ill-formed with no diagnostic
 *   required.
 *
 * @li Otherwise, <tt>set_value(R, Vs...)</tt>, if that expression is valid,
 *   with overload resolution performed in a context that includes the
 *   declaration <tt>void set_value();</tt> and that does not include a
 *   declaration of <tt>execution::set_value</tt>. If the function selected by
 *   overload resolution does not send the value(s) <tt>Vs...</tt> to the
 *   receiver <tt>R</tt>'s value channel, the program is ill-formed with no
 *   diagnostic required.
 *
 * @li Otherwise, <tt>execution::set_value(R, Vs...)</tt> is ill-formed.
 */
inline constexpr unspecified set_value = unspecified;

/// A type trait that determines whether a @c set_value expression is
/// well-formed.
/**
 * Class template @c can_set_value is a trait that is derived from
 * @c true_type if the expression <tt>execution::set_value(std::declval<R>(),
 * std::declval<Vs>()...)</tt> is well formed; otherwise @c false_type.
 */
template <typename R, typename... Vs>
struct can_set_value :
  integral_constant<bool, automatically_determined>
{
};

} // namespace execution
} // namespace asio

#else // defined(GENERATING_DOCUMENTATION)

namespace asio_execution_set_value_fn {

using asio::decay_t;
using asio::declval;
using asio::enable_if_t;
using asio::traits::set_value_free;
using asio::traits::set_value_member;

void set_value();

enum overload_type
{
  call_member,
  call_free,
  ill_formed
};

template <typename R, typename Vs, typename = void, typename = void>
struct call_traits
{
  static constexpr overload_type overload = ill_formed;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
};

template <typename R, typename Vs>
struct call_traits<R, Vs,
  enable_if_t<
    set_value_member<R, Vs>::is_valid
  >> :
  set_value_member<R, Vs>
{
  static constexpr overload_type overload = call_member;
};

template <typename R, typename Vs>
struct call_traits<R, Vs,
  enable_if_t<
    !set_value_member<R, Vs>::is_valid
  >,
  enable_if_t<
    set_value_free<R, Vs>::is_valid
  >> :
  set_value_free<R, Vs>
{
  static constexpr overload_type overload = call_free;
};

struct impl
{
  template <typename R, typename... Vs>
  constexpr enable_if_t<
    call_traits<R, void(Vs...)>::overload == call_member,
    typename call_traits<R, void(Vs...)>::result_type
  >
  operator()(R&& r, Vs&&... v) const
    noexcept(call_traits<R, void(Vs...)>::is_noexcept)
  {
    return static_cast<R&&>(r).set_value(static_cast<Vs&&>(v)...);
  }

  template <typename R, typename... Vs>
  constexpr enable_if_t<
    call_traits<R, void(Vs...)>::overload == call_free,
    typename call_traits<R, void(Vs...)>::result_type
  >
  operator()(R&& r, Vs&&... v) const
    noexcept(call_traits<R, void(Vs...)>::is_noexcept)
  {
    return set_value(static_cast<R&&>(r), static_cast<Vs&&>(v)...);
  }
};

template <typename T = impl>
struct static_instance
{
  static const T instance;
};

template <typename T>
const T static_instance<T>::instance = {};

} // namespace asio_execution_set_value_fn
namespace asio {
namespace execution {
namespace {

static constexpr const asio_execution_set_value_fn::impl&
  set_value = asio_execution_set_value_fn::static_instance<>::instance;

} // namespace

template <typename R, typename... Vs>
struct can_set_value :
  integral_constant<bool,
    asio_execution_set_value_fn::call_traits<R, void(Vs...)>::overload !=
      asio_execution_set_value_fn::ill_formed>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename R, typename... Vs>
constexpr bool can_set_value_v = can_set_value<R, Vs...>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename R, typename... Vs>
struct is_nothrow_set_value :
  integral_constant<bool,
    asio_execution_set_value_fn::call_traits<R, void(Vs...)>::is_noexcept>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename R, typename... Vs>
constexpr bool is_nothrow_set_value_v = is_nothrow_set_value<R, Vs...>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

} // namespace execution
} // namespace asio

#endif // defined(GENERATING_DOCUMENTATION)

#include "asio/detail/pop_options.hpp"

#endif // !defined(ASIO_NO_DEPRECATED)

#endif // ASIO_EXECUTION_SET_VALUE_HPP
