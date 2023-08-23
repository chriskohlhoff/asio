//
// execution/receiver.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_RECEIVER_HPP
#define ASIO_EXECUTION_RECEIVER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_DEPRECATED)

#include <exception>
#include "asio/detail/type_traits.hpp"
#include "asio/execution/set_done.hpp"
#include "asio/execution/set_error.hpp"
#include "asio/execution/set_value.hpp"


#if defined(ASIO_HAS_DEDUCED_SET_DONE_FREE_TRAIT) \
  && defined(ASIO_HAS_DEDUCED_SET_DONE_MEMBER_TRAIT) \
  && defined(ASIO_HAS_DEDUCED_SET_ERROR_FREE_TRAIT) \
  && defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT) \
  && defined(ASIO_HAS_DEDUCED_SET_VALUE_FREE_TRAIT) \
  && defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT) \
  && defined(ASIO_HAS_DEDUCED_RECEIVER_OF_FREE_TRAIT) \
  && defined(ASIO_HAS_DEDUCED_RECEIVER_OF_MEMBER_TRAIT)
# define ASIO_HAS_DEDUCED_EXECUTION_IS_RECEIVER_TRAIT 1
#endif // defined(ASIO_HAS_DEDUCED_SET_DONE_FREE_TRAIT)
       //   && defined(ASIO_HAS_DEDUCED_SET_DONE_MEMBER_TRAIT)
       //   && defined(ASIO_HAS_DEDUCED_SET_ERROR_FREE_TRAIT)
       //   && defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)
       //   && defined(ASIO_HAS_DEDUCED_SET_VALUE_FREE_TRAIT)
       //   && defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)
       //   && defined(ASIO_HAS_DEDUCED_RECEIVER_OF_FREE_TRAIT)
       //   && defined(ASIO_HAS_DEDUCED_RECEIVER_OF_MEMBER_TRAIT)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename T, typename E>
struct is_receiver_base :
  integral_constant<bool,
    is_move_constructible<remove_cvref_t<T>>::value
      && is_constructible<remove_cvref_t<T>, T>::value
  >
{
};

} // namespace detail

#define ASIO_EXECUTION_RECEIVER_ERROR_DEFAULT = std::exception_ptr

/// The is_receiver trait detects whether a type T satisfies the
/// execution::receiver concept.
/**
 * Class template @c is_receiver is a type trait that is derived from @c
 * true_type if the type @c T meets the concept definition for a receiver for
 * error type @c E, otherwise @c false_type.
 */
template <typename T, typename E ASIO_EXECUTION_RECEIVER_ERROR_DEFAULT>
struct is_receiver :
#if defined(GENERATING_DOCUMENTATION)
  integral_constant<bool, automatically_determined>
#else // defined(GENERATING_DOCUMENTATION)
  conditional<
    can_set_done<remove_cvref_t<T>>::value
      && is_nothrow_set_done<remove_cvref_t<T>>::value
      && can_set_error<remove_cvref_t<T>, E>::value
      && is_nothrow_set_error<remove_cvref_t<T>, E>::value,
    detail::is_receiver_base<T, E>,
    false_type
  >::type
#endif // defined(GENERATING_DOCUMENTATION)
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename E ASIO_EXECUTION_RECEIVER_ERROR_DEFAULT>
constexpr const bool is_receiver_v = is_receiver<T, E>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

#if defined(ASIO_HAS_CONCEPTS)

template <typename T, typename E ASIO_EXECUTION_RECEIVER_ERROR_DEFAULT>
ASIO_CONCEPT receiver = is_receiver<T, E>::value;

#define ASIO_EXECUTION_RECEIVER ::asio::execution::receiver

#else // defined(ASIO_HAS_CONCEPTS)

#define ASIO_EXECUTION_RECEIVER typename

#endif // defined(ASIO_HAS_CONCEPTS)

/// The is_receiver_of trait detects whether a type T satisfies the
/// execution::receiver_of concept for some set of value arguments.
/**
 * Class template @c is_receiver_of is a type trait that is derived from @c
 * true_type if the type @c T meets the concept definition for a receiver for
 * value arguments @c Vs, otherwise @c false_type.
 */
template <typename T, typename... Vs>
struct is_receiver_of :
#if defined(GENERATING_DOCUMENTATION)
  integral_constant<bool, automatically_determined>
#else // defined(GENERATING_DOCUMENTATION)
  conditional<
    is_receiver<T>::value,
    can_set_value<remove_cvref_t<T>, Vs...>,
    false_type
  >::type
#endif // defined(GENERATING_DOCUMENTATION)
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename... Vs>
constexpr const bool is_receiver_of_v =
  is_receiver_of<T, Vs...>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

#if defined(ASIO_HAS_CONCEPTS)

template <typename T, typename... Vs>
ASIO_CONCEPT receiver_of = is_receiver_of<T, Vs...>::value;

#define ASIO_EXECUTION_RECEIVER_OF_0 \
  ::asio::execution::receiver_of

#define ASIO_EXECUTION_RECEIVER_OF_1(v) \
  ::asio::execution::receiver_of<v>

#else // defined(ASIO_HAS_CONCEPTS)

#define ASIO_EXECUTION_RECEIVER_OF_0 typename
#define ASIO_EXECUTION_RECEIVER_OF_1(v) typename

#endif // defined(ASIO_HAS_CONCEPTS)

/// The is_nothrow_receiver_of trait detects whether a type T satisfies the
/// execution::receiver_of concept for some set of value arguments, with a
/// noexcept @c set_value operation.
/**
 * Class template @c is_nothrow_receiver_of is a type trait that is derived
 * from @c true_type if the type @c T meets the concept definition for a
 * receiver for value arguments @c Vs, and the expression
 * <tt>execution::set_value(declval<T>(), declval<Ts>()...)</tt> is noexcept,
 * otherwise @c false_type.
 */
template <typename T, typename... Vs>
struct is_nothrow_receiver_of :
#if defined(GENERATING_DOCUMENTATION)
  integral_constant<bool, automatically_determined>
#else // defined(GENERATING_DOCUMENTATION)
  integral_constant<bool,
    is_receiver_of<T, Vs...>::value
      && is_nothrow_set_value<remove_cvref_t<T>, Vs...>::value
  >
#endif // defined(GENERATING_DOCUMENTATION)
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename... Vs>
constexpr const bool is_nothrow_receiver_of_v =
  is_nothrow_receiver_of<T, Vs...>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(ASIO_NO_DEPRECATED)

#endif // ASIO_EXECUTION_RECEIVER_HPP
