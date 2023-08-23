//
// traits/set_done_free.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TRAITS_SET_DONE_FREE_HPP
#define ASIO_TRAITS_SET_DONE_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_SET_DONE_FREE_TRAIT 1
#endif // defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename T, typename = void>
struct set_done_free_default;

template <typename T, typename = void>
struct set_done_free;

} // namespace traits
namespace detail {

struct no_set_done_free
{
  static constexpr bool is_valid = false;
  static constexpr bool is_noexcept = false;
};

#if defined(ASIO_HAS_DEDUCED_SET_DONE_FREE_TRAIT)

template <typename T, typename = void>
struct set_done_free_trait : no_set_done_free
{
};

template <typename T>
struct set_done_free_trait<T,
  void_t<
    decltype(set_done(declval<T>()))
  >>
{
  static constexpr bool is_valid = true;

  using result_type = decltype(set_done(declval<T>()));

  static constexpr bool is_noexcept = noexcept(set_done(declval<T>()));
};

#else // defined(ASIO_HAS_DEDUCED_SET_DONE_FREE_TRAIT)

template <typename T, typename = void>
struct set_done_free_trait :
  conditional_t<
    is_same<T, remove_reference_t<T>>::value,
    conditional_t<
      is_same<T, add_const_t<T>>::value,
      no_set_done_free,
      traits::set_done_free<add_const_t<T>>
    >,
    traits::set_done_free<remove_reference_t<T>>
  >
{
};

#endif // defined(ASIO_HAS_DEDUCED_SET_DONE_FREE_TRAIT)

} // namespace detail
namespace traits {

template <typename T, typename>
struct set_done_free_default :
  detail::set_done_free_trait<T>
{
};

template <typename T, typename>
struct set_done_free :
  set_done_free_default<T>
{
};

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TRAITS_SET_DONE_FREE_HPP
