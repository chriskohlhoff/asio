//
// execution/executor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_EXECUTOR_HPP
#define ASIO_EXECUTION_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/invocable_archetype.hpp"
#include "asio/traits/equality_comparable.hpp"
#include "asio/traits/execute_member.hpp"

#if !defined(ASIO_NO_DEPRECATED)
# include "asio/execution/execute.hpp"
#endif // !defined(ASIO_NO_DEPRECATED)

#if defined(ASIO_HAS_DEDUCED_EXECUTE_FREE_TRAIT) \
  && defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT) \
  && defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)
# define ASIO_HAS_DEDUCED_EXECUTION_IS_EXECUTOR_TRAIT 1
#endif // defined(ASIO_HAS_DEDUCED_EXECUTE_FREE_TRAIT)
       //   && defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)
       //   && defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename T, typename F,
    typename = void, typename = void, typename = void, typename = void,
    typename = void, typename = void, typename = void, typename = void>
struct is_executor_of_impl : false_type
{
};

template <typename T, typename F>
struct is_executor_of_impl<T, F,
#if defined(ASIO_NO_DEPRECATED)
  enable_if_t<
    traits::execute_member<add_const_t<T>, F>::is_valid
  >,
#else // defined(ASIO_NO_DEPRECATED)
  enable_if_t<
    can_execute<add_const_t<T>, F>::value
  >,
#endif // defined(ASIO_NO_DEPRECATED)
  void_t<
    result_of_t<decay_t<F>&()>
  >,
  enable_if_t<
    is_constructible<decay_t<F>, F>::value
  >,
  enable_if_t<
    is_move_constructible<decay_t<F>>::value
  >,
  enable_if_t<
    is_nothrow_copy_constructible<T>::value
  >,
  enable_if_t<
    is_nothrow_destructible<T>::value
  >,
  enable_if_t<
    traits::equality_comparable<T>::is_valid
  >,
  enable_if_t<
    traits::equality_comparable<T>::is_noexcept
  >> : true_type
{
};

template <typename T, typename = void>
struct executor_shape
{
  typedef std::size_t type;
};

template <typename T>
struct executor_shape<T, void_t<typename T::shape_type>>
{
  typedef typename T::shape_type type;
};

template <typename T, typename Default, typename = void>
struct executor_index
{
  typedef Default type;
};

template <typename T, typename Default>
struct executor_index<T, Default, void_t<typename T::index_type>>
{
  typedef typename T::index_type type;
};

} // namespace detail

/// The is_executor trait detects whether a type T satisfies the
/// execution::executor concept.
/**
 * Class template @c is_executor is a UnaryTypeTrait that is derived from @c
 * true_type if the type @c T meets the concept definition for an executor,
 * otherwise @c false_type.
 */
template <typename T>
struct is_executor :
#if defined(GENERATING_DOCUMENTATION)
  integral_constant<bool, automatically_determined>
#else // defined(GENERATING_DOCUMENTATION)
  detail::is_executor_of_impl<T, invocable_archetype>
#endif // defined(GENERATING_DOCUMENTATION)
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T>
constexpr const bool is_executor_v = is_executor<T>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

#if defined(ASIO_HAS_CONCEPTS)

template <typename T>
ASIO_CONCEPT executor = is_executor<T>::value;

#define ASIO_EXECUTION_EXECUTOR ::asio::execution::executor

#else // defined(ASIO_HAS_CONCEPTS)

#define ASIO_EXECUTION_EXECUTOR typename

#endif // defined(ASIO_HAS_CONCEPTS)

/// The is_executor_of trait detects whether a type T satisfies the
/// execution::executor_of concept for some set of value arguments.
/**
 * Class template @c is_executor_of is a type trait that is derived from @c
 * true_type if the type @c T meets the concept definition for an executor
 * that is invocable with a function object of type @c F, otherwise @c
 * false_type.
 */
template <typename T, typename F>
struct is_executor_of :
#if defined(GENERATING_DOCUMENTATION)
  integral_constant<bool, automatically_determined>
#else // defined(GENERATING_DOCUMENTATION)
  integral_constant<bool,
    is_executor<T>::value && detail::is_executor_of_impl<T, F>::value
  >
#endif // defined(GENERATING_DOCUMENTATION)
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename F>
constexpr const bool is_executor_of_v =
  is_executor_of<T, F>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

#if defined(ASIO_HAS_CONCEPTS)

template <typename T, typename F>
ASIO_CONCEPT executor_of = is_executor_of<T, F>::value;

#define ASIO_EXECUTION_EXECUTOR_OF(f) \
  ::asio::execution::executor_of<f>

#else // defined(ASIO_HAS_CONCEPTS)

#define ASIO_EXECUTION_EXECUTOR_OF typename

#endif // defined(ASIO_HAS_CONCEPTS)

/// The executor_shape trait detects the type used by an executor to represent
/// the shape of a bulk operation.
/**
 * Class template @c executor_shape is a type trait with a nested type alias
 * @c type whose type is @c T::shape_type if @c T::shape_type is valid,
 * otherwise @c std::size_t.
 */
template <typename T>
struct executor_shape
#if !defined(GENERATING_DOCUMENTATION)
  : detail::executor_shape<T>
#endif // !defined(GENERATING_DOCUMENTATION)
{
#if defined(GENERATING_DOCUMENTATION)
 /// @c T::shape_type if @c T::shape_type is valid, otherwise @c std::size_t.
 typedef automatically_determined type;
#endif // defined(GENERATING_DOCUMENTATION)
};

template <typename T>
using executor_shape_t = typename executor_shape<T>::type;

/// The executor_index trait detects the type used by an executor to represent
/// an index within a bulk operation.
/**
 * Class template @c executor_index is a type trait with a nested type alias
 * @c type whose type is @c T::index_type if @c T::index_type is valid,
 * otherwise @c executor_shape_t<T>.
 */
template <typename T>
struct executor_index
#if !defined(GENERATING_DOCUMENTATION)
  : detail::executor_index<T, typename executor_shape<T>::type>
#endif // !defined(GENERATING_DOCUMENTATION)
{
#if defined(GENERATING_DOCUMENTATION)
 /// @c T::index_type if @c T::index_type is valid, otherwise
 /// @c executor_shape_t<T>.
 typedef automatically_determined type;
#endif // defined(GENERATING_DOCUMENTATION)
};

template <typename T>
using executor_index_t = typename executor_index<T>::type;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_EXECUTOR_HPP
