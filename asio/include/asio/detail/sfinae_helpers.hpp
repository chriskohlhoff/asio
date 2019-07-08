//
// detail/sfinae_helpers.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2019 Alexander Karzhenkov
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SFINAE_HELPERS_HPP
#define ASIO_DETAIL_SFINAE_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename C>
struct sfinae_result_helper : C
{
  using C::detector;

  template <typename>
  static char(&detector(...))[2];
};

template <typename C, typename T>
struct sfinae_result
  : integral_constant<bool,
      sizeof(sfinae_result_helper<C>::template detector<T>(0)) == 1>
{
};

template <typename C>
struct sfinae_result<C, void> : false_type
{
};

struct sfinae_check_base
{
  template <int> struct result { };

  template <typename To, typename From>
  static typename enable_if<
    is_convertible<From, To>::value>::type
  is_convertible_to(From);

  template <typename T>
  static void check();

  template <bool C>
  static typename enable_if<C>::type
  check();

  template <typename A, typename B>
  static typename enable_if<is_same<A, B>::value>::type
  is_same_as(B);

};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SFINAE_HELPERS_HPP
