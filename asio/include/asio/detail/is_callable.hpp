//
// detail/is_callable.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IS_CALLABLE_HPP
#define ASIO_DETAIL_IS_CALLABLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

namespace asio {
namespace detail {

struct is_callable_big_type
{
  char data[10];
};

struct is_callable_base
{
  void operator()() {}
};

template <typename T>
struct is_callable_derived
  : T, is_callable_base
{
};

template <typename T, T>
struct is_callable_check
{
};

template <typename>
char is_callable_class_helper(...);

template <typename T>
is_callable_big_type is_callable_class_helper(
    is_callable_check<void (is_callable_base::*)(),
      &is_callable_derived<T>::operator()>*);

template <typename T>
struct is_callable_class
  : integral_constant<bool,
      sizeof(is_callable_class_helper<T>(0)) == 1>
{
};

template <typename T>
struct is_callable_function
  : is_function<typename remove_pointer<
      typename remove_reference<T>::type>::type>
{
};

template <typename T>
struct is_callable
  : conditional<is_class<T>::value,
      is_callable_class<T>,
      is_callable_function<T> >::type
{
};

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_IS_CALLABLE_HPP
