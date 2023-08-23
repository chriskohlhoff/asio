//
// execution/schedule.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_SCHEDULE_HPP
#define ASIO_EXECUTION_SCHEDULE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_DEPRECATED)

#include "asio/detail/type_traits.hpp"
#include "asio/execution/executor.hpp"
#include "asio/traits/schedule_member.hpp"
#include "asio/traits/schedule_free.hpp"

#include "asio/detail/push_options.hpp"

#if defined(GENERATING_DOCUMENTATION)

namespace asio {
namespace execution {

/// A customisation point that is used to obtain a sender from a scheduler.
/**
 * The name <tt>execution::schedule</tt> denotes a customisation point object.
 * For some subexpression <tt>s</tt>, let <tt>S</tt> be a type such that
 * <tt>decltype((s))</tt> is <tt>S</tt>. The expression
 * <tt>execution::schedule(s)</tt> is expression-equivalent to:
 *
 * @li <tt>s.schedule()</tt>, if that expression is valid and its type models
 *   <tt>sender</tt>.
 *
 * @li Otherwise, <tt>schedule(s)</tt>, if that expression is valid and its
 *   type models <tt>sender</tt> with overload resolution performed in a context
 *   that includes the declaration <tt>void schedule();</tt> and that does not
 *   include a declaration of <tt>execution::schedule</tt>.
 *
 * @li Otherwise, <tt>S</tt> if <tt>S</tt> satisfies <tt>executor</tt>.
 *
 * @li Otherwise, <tt>execution::schedule(s)</tt> is ill-formed.
 */
inline constexpr unspecified schedule = unspecified;

/// A type trait that determines whether a @c schedule expression is
/// well-formed.
/**
 * Class template @c can_schedule is a trait that is derived from @c true_type
 * if the expression <tt>execution::schedule(std::declval<S>())</tt> is well
 * formed; otherwise @c false_type.
 */
template <typename S>
struct can_schedule :
  integral_constant<bool, automatically_determined>
{
};

} // namespace execution
} // namespace asio

#else // defined(GENERATING_DOCUMENTATION)

namespace asio_execution_schedule_fn {

using asio::decay_t;
using asio::declval;
using asio::enable_if_t;
using asio::execution::is_executor;
using asio::traits::schedule_free;
using asio::traits::schedule_member;

void schedule();

enum overload_type
{
  identity,
  call_member,
  call_free,
  ill_formed
};

template <typename S, typename = void, typename = void, typename = void>
struct call_traits
{
  static constexpr overload_type overload = ill_formed;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
};

template <typename S>
struct call_traits<S,
  enable_if_t<
    schedule_member<S>::is_valid
  >> :
  schedule_member<S>
{
  static constexpr overload_type overload = call_member;
};

template <typename S>
struct call_traits<S,
  enable_if_t<
    !schedule_member<S>::is_valid
  >,
  enable_if_t<
    schedule_free<S>::is_valid
  >> :
  schedule_free<S>
{
  static constexpr overload_type overload = call_free;
};

template <typename S>
struct call_traits<S,
  enable_if_t<
    !schedule_member<S>::is_valid
  >,
  enable_if_t<
    !schedule_free<S>::is_valid
  >,
  enable_if_t<
    is_executor<decay_t<S>>::value
  >>
{
  static constexpr overload_type overload = identity;
  static constexpr bool is_noexcept = true;

  typedef S&& result_type;
};

struct impl
{
  template <typename S>
  constexpr enable_if_t<
    call_traits<S>::overload == identity,
    typename call_traits<S>::result_type
  >
  operator()(S&& s) const
    noexcept(call_traits<S>::is_noexcept)
  {
    return static_cast<S&&>(s);
  }

  template <typename S>
  constexpr enable_if_t<
    call_traits<S>::overload == call_member,
    typename call_traits<S>::result_type
  >
  operator()(S&& s) const
    noexcept(call_traits<S>::is_noexcept)
  {
    return static_cast<S&&>(s).schedule();
  }

  template <typename S>
  constexpr enable_if_t<
    call_traits<S>::overload == call_free,
    typename call_traits<S>::result_type
  >
  operator()(S&& s) const
    noexcept(call_traits<S>::is_noexcept)
  {
    return schedule(static_cast<S&&>(s));
  }
};

template <typename T = impl>
struct static_instance
{
  static const T instance;
};

template <typename T>
const T static_instance<T>::instance = {};

} // namespace asio_execution_schedule_fn
namespace asio {
namespace execution {
namespace {

static constexpr const asio_execution_schedule_fn::impl&
  schedule = asio_execution_schedule_fn::static_instance<>::instance;

} // namespace

template <typename S>
struct can_schedule :
  integral_constant<bool,
    asio_execution_schedule_fn::call_traits<S>::overload !=
      asio_execution_schedule_fn::ill_formed>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S>
constexpr bool can_schedule_v = can_schedule<S>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S>
struct is_nothrow_schedule :
  integral_constant<bool,
    asio_execution_schedule_fn::call_traits<S>::is_noexcept>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S>
constexpr bool is_nothrow_schedule_v = is_nothrow_schedule<S>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

} // namespace execution
} // namespace asio

#endif // defined(GENERATING_DOCUMENTATION)

#include "asio/detail/pop_options.hpp"

#endif // !defined(ASIO_NO_DEPRECATED)

#endif // ASIO_EXECUTION_SCHEDULE_HPP
