//
// detail/result_type.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_RESULT_TYPE_HPP
#define ASIO_DETAIL_RESULT_TYPE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"

namespace asio {
namespace detail {

template <typename T>
struct result_type_check
{
  typedef void type;
};

template <typename R>
struct result_type_base
{
  typedef R type;
};

template <typename T>
struct result_type_function {};

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename R, typename... Args>
struct result_type_function<R(Args...)>
  : result_type_base<R> {};

template <typename R, typename... Args>
struct result_type_function<R(*)(Args...)>
  : result_type_base<R> {};

template <typename R, typename... Args>
struct result_type_function<R(&)(Args...)>
  : result_type_base<R> {};

template <typename R, typename T, typename... Args>
struct result_type_function<R(T::*)(Args...)>
  : result_type_base<R> {};

template <typename R, typename T, typename... Args>
struct result_type_function<R(T::*)(Args...) const>
  : result_type_base<R> {};

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename R>
struct result_type_function<R()>
  : result_type_base<R> {};

template <typename R>
struct result_type_function<R(*)()>
  : result_type_base<R> {};

template <typename R>
struct result_type_function<R(&)()>
  : result_type_base<R> {};

template <typename R, typename T>
struct result_type_function<R(T::*)()>
  : result_type_base<R> {};

template <typename R, typename T>
struct result_type_function<R(T::*)() const>
  : result_type_base<R> {};

#define ASIO_PRIVATE_RESULT_TYPE_FUNCTION(n) \
  \
  template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
  struct result_type_function<R(ASIO_VARIADIC_TARGS(n))> \
    : result_type_base<R> {}; \
  \
  template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
  struct result_type_function<R(*)(ASIO_VARIADIC_TARGS(n))> \
    : result_type_base<R> {}; \
  \
  template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
  struct result_type_function<R(&)(ASIO_VARIADIC_TARGS(n))> \
    : result_type_base<R> {}; \
  \
  template <typename R, typename T, ASIO_VARIADIC_TPARAMS(n)> \
  struct result_type_function<R(T::*)(ASIO_VARIADIC_TARGS(n))> \
    : result_type_base<R> {}; \
  \
  template <typename R, typename T, ASIO_VARIADIC_TPARAMS(n)> \
  struct result_type_function<R(T::*)(ASIO_VARIADIC_TARGS(n)) const> \
    : result_type_base<R> {}; \
  /**/

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_RESULT_TYPE_FUNCTION)

#undef ASIO_PRIVATE_RESULT_TYPE_FUNCTION

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename T, typename = void>
struct result_type_using_nested_typedef {};

template <typename T>
struct result_type_using_nested_typedef<T,
  typename result_type_check<typename T::result_type>::type>
    : result_type_base<typename T::result_type> {};

template <typename T, typename = void>
struct result_type_class
  : result_type_using_nested_typedef<T> {};

#if defined(ASIO_HAS_DECLTYPE)

template <typename T>
struct result_type_class<T,
  typename result_type_check<decltype(&T::operator())>::type>
    : result_type_function<decltype(&T::operator())> {};

#endif // defined(ASIO_HAS_DECLTYPE)

template <typename T>
struct result_type
  : conditional<is_class<T>::value,
      result_type_class<T>,
      result_type_function<T> >::type {};

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_RESULT_TYPE_HPP
