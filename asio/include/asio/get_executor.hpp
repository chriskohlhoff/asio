//
// get_executor.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_GET_EXECUTOR_HPP
#define ASIO_GET_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/is_callable.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/unspecified_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename>
struct get_executor_check
{
  typedef void type;
};

template <typename T, typename = void>
struct get_unspecified_executor_impl {};

template <typename T>
struct get_unspecified_executor_impl<T,
  typename enable_if<is_callable<T>::value>::type>
{
  typedef unspecified_executor type;

  static type get(const T&)
  {
    return type();
  }
};

template <typename T, typename = void>
struct get_executor_impl
  : get_unspecified_executor_impl<T> {};

template <typename T>
struct get_executor_impl<T,
  typename get_executor_check<typename T::executor_type>::type>
{
  typedef typename T::executor_type type;

  static type get(const T& t)
  {
    return t.get_executor();
  }
};

} // namespace detail

/// Default hook to obtain the executor associated with an object.
/**
 * The @c get_executor function returns an object's associated executor. For
 * function objects, this is the executor that would be used to invoke the
 * given function. This default implementation behaves as follows:
 *
 * @li if the object type has a nested type @c executor_type, returns the
 * result of the object's @c get_executor() member function;
 *
 * @li if the object is callable, returns an @c unspecified_executor object;
 *
 * @li otherwise, this function does not participate in overload resolution.
 *
 * To customise the executor associated with a function object type, you may
 * use either of the following approaches:
 *
 * @li Provide a nested typedef @c executor_type and a const member function
 * @c get_executor().
 *
 * @li Create an overload of the free function @c get_executor() for the
 * function object type.
 *
 * The Asio library implementation, and any other code that calls the @c
 * get_executor() function, should use argument-dependent lookup to locate the
 * correct @c get_executor() overload as follows:
 *
 * @code using asio::get_executor;
 * auto ex = get_executor(my_function); @endcode
 *
 * For C++03, where automatic type deduction is not available, programs must
 * utilise the asio::get_executor_type type trait to determine the correct
 * executor type:
 *
 * @code typedef asio::get_executor_type<T>::type executor_type;
 * using asio::get_executor;
 * executor_type ex = get_executor(my_function); @endcode
 */
#if defined(GENERATING_DOCUMENTATION)
template <typename T>
see_below get_executor(const T&);
#else // defined(GENERATING_DOCUMENTATION)
template <typename T>
inline typename detail::get_executor_impl<T>::type get_executor(const T& t)
{
  return detail::get_executor_impl<T>::get(t);
}
#endif // defined(GENERATING_DOCUMENTATION)

/// Type trait used to determine associated executor type.
/**
 * You may specialise @c get_executor_type for your own function objects and
 * handlers to customise the executor type that will be used to invoke them.
 */
template <typename T, typename = void>
struct get_executor_type
{
#if defined(ASIO_HAS_DECLTYPE)
private:
  static T val();
public:
  typedef decltype(get_executor((get_executor_type::val)())) type;
#else // defined(ASIO_HAS_DECLTYPE)
  typedef typename detail::get_executor_impl<T>::type type;
#endif // defined(ASIO_HAS_DECLTYPE)
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_GET_EXECUTOR_HPP
