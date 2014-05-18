//
// get_allocator.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_GET_ALLOCATOR_HPP
#define ASIO_GET_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/is_callable.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/unspecified_allocator.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename>
struct get_allocator_check
{
  typedef void type;
};

template <typename T, typename = void>
struct get_unspecified_allocator_impl {};

template <typename T>
struct get_unspecified_allocator_impl<T,
  typename enable_if<is_callable<T>::value>::type>
{
  typedef unspecified_allocator<void> type;

  static type get(const T&) ASIO_NOEXCEPT
  {
    return type();
  }
};

template <typename T, typename = void>
struct get_allocator_impl
  : get_unspecified_allocator_impl<T> {};

template <typename T>
struct get_allocator_impl<T,
  typename get_allocator_check<typename T::allocator_type>::type>
{
  typedef typename T::allocator_type type;

  static type get(const T& t) ASIO_NOEXCEPT
  {
    return t.get_allocator();
  }
};

} // namespace detail

/// Default hook to obtain the allocator associated with an object.
/**
 * The @c get_allocator function returns an object's associated allocator. For
 * function objects, this is the allocator that would be used to allocate any
 * memory associated with the function's eventual invocation by an executor.
 * This default implementation behaves as follows:
 *
 * @li if the object type has a nested type @c allocator_type, returns the
 * result of the object's @c get_allocator() member function;
 *
 * @li if the object is callable, returns a @c std::allocator<void> object;
 *
 * @li otherwise, this function does not participate in overload resolution.
 *
 * To customise the allocator associated with a function object type, you may
 * use either of the following approaches:
 *
 * @li Provide a nested typedef @c allocator_type and a const member function
 * @c get_allocator().
 *
 * @li Create an overload of the free function @c get_allocator() for the
 * function object type.
 *
 * The Asio library implementation, and any other code that calls the @c
 * get_allocator() function, should use argument-dependent lookup to locate the
 * correct @c get_allocator() overload as follows:
 *
 * @code using asio::get_allocator;
 * auto ex = get_allocator(my_function); @endcode
 *
 * For C++03, where automatic type deduction is not available, programs must
 * utilise the asio::get_allocator_type type trait to determine the correct
 * allocator type:
 *
 * @code typedef asio::get_allocator_type<T>::type allocator_type;
 * using asio::get_allocator;
 * allocator_type ex = get_allocator(my_function); @endcode
 */
#if defined(GENERATING_DOCUMENTATION)
template <typename T>
see_below get_allocator(const T&) noexcept;
#else // defined(GENERATING_DOCUMENTATION)
template <typename T>
inline typename detail::get_allocator_impl<T>::type
get_allocator(const T& t) ASIO_NOEXCEPT
{
  return detail::get_allocator_impl<T>::get(t);
}
#endif // defined(GENERATING_DOCUMENTATION)

/// Type trait used to determine associated allocator type.
/**
 * You may specialise @c get_allocator_type for your own function objects and
 * handlers to customise the allocator type that will be used to allocate memory
 * associated with the function and its eventual invocation by an executor.
 */
template <typename T, typename = void>
struct get_allocator_type
{
#if defined(ASIO_HAS_DECLTYPE)
private:
  static T val();
public:
  typedef decltype(get_allocator((get_allocator_type::val)())) type;
#else // defined(ASIO_HAS_DECLTYPE)
  typedef typename detail::get_allocator_impl<T>::type type;
#endif // defined(ASIO_HAS_DECLTYPE)
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_GET_ALLOCATOR_HPP
