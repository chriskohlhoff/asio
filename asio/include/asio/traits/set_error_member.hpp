//
// traits/set_error_member.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TRAITS_SET_ERROR_MEMBER_HPP
#define ASIO_TRAITS_SET_ERROR_MEMBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT 1
#endif // defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename T, typename E, typename = void>
struct set_error_member_default;

template <typename T, typename E, typename = void>
struct set_error_member;

} // namespace traits
namespace detail {

struct no_set_error_member
{
  static constexpr bool is_valid = false;
  static constexpr bool is_noexcept = false;
};

#if defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)

template <typename T, typename E, typename = void>
struct set_error_member_trait : no_set_error_member
{
};

template <typename T, typename E>
struct set_error_member_trait<T, E,
  void_t<
    decltype(declval<T>().set_error(declval<E>()))
  >>
{
  static constexpr bool is_valid = true;

  using result_type = decltype(
    declval<T>().set_error(declval<E>()));

  static constexpr bool is_noexcept =
    noexcept(declval<T>().set_error(declval<E>()));
};

#else // defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)

template <typename T, typename E, typename = void>
struct set_error_member_trait :
  conditional_t<
    is_same<T, remove_reference_t<T>>::value
      && is_same<E, decay_t<E>>::value,
    conditional_t<
      is_same<T, add_const_t<T>>::value,
      no_set_error_member,
      traits::set_error_member<add_const_t<T>, E>
    >,
    traits::set_error_member<
      remove_reference_t<T>,
      decay_t<E>>
  >
{
};

#endif // defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)

} // namespace detail
namespace traits {

template <typename T, typename E, typename>
struct set_error_member_default :
  detail::set_error_member_trait<T, E>
{
};

template <typename T, typename E, typename>
struct set_error_member :
  set_error_member_default<T, E>
{
};

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TRAITS_SET_ERROR_MEMBER_HPP
