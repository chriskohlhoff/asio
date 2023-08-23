//
// traits/set_value_member.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TRAITS_SET_VALUE_MEMBER_HPP
#define ASIO_TRAITS_SET_VALUE_MEMBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT 1
#endif // defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename T, typename Vs, typename = void>
struct set_value_member_default;

template <typename T, typename Vs, typename = void>
struct set_value_member;

} // namespace traits
namespace detail {

struct no_set_value_member
{
  static constexpr bool is_valid = false;
  static constexpr bool is_noexcept = false;
};

#if defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

template <typename T, typename Vs, typename = void>
struct set_value_member_trait : no_set_value_member
{
};

template <typename T, typename... Vs>
struct set_value_member_trait<T, void(Vs...),
  void_t<
    decltype(declval<T>().set_value(declval<Vs>()...))
  >>
{
  static constexpr bool is_valid = true;

  using result_type = decltype(
    declval<T>().set_value(declval<Vs>()...));

  static constexpr bool is_noexcept =
    noexcept(declval<T>().set_value(declval<Vs>()...));
};

#else // defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

template <typename T, typename Vs, typename = void>
struct set_value_member_trait;

template <typename T, typename... Vs>
struct set_value_member_trait<T, void(Vs...)> :
  conditional_t<
    is_same<T, remove_reference_t<T>>::value
      && conjunction<is_same<Vs, decay_t<Vs>>...>::value,
    conditional_t<
      is_same<T, add_const_t<T>>::value,
      no_set_value_member,
      traits::set_value_member<add_const_t<T>, void(Vs...)>
    >,
    traits::set_value_member<
      remove_reference_t<T>,
      void(decay_t<Vs>...)>
  >
{
};

#endif // defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

} // namespace detail
namespace traits {

template <typename T, typename Vs, typename>
struct set_value_member_default :
  detail::set_value_member_trait<T, Vs>
{
};

template <typename T, typename Vs, typename>
struct set_value_member :
  set_value_member_default<T, Vs>
{
};

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TRAITS_SET_VALUE_MEMBER_HPP
