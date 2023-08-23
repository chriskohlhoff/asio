//
// traits/connect_member.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TRAITS_CONNECT_MEMBER_HPP
#define ASIO_TRAITS_CONNECT_MEMBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_CONNECT_MEMBER_TRAIT 1
#endif // defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename S, typename R, typename = void>
struct connect_member_default;

template <typename S, typename R, typename = void>
struct connect_member;

} // namespace traits
namespace detail {

struct no_connect_member
{
  static constexpr bool is_valid = false;
  static constexpr bool is_noexcept = false;
};

#if defined(ASIO_HAS_DEDUCED_CONNECT_MEMBER_TRAIT)

template <typename S, typename R, typename = void>
struct connect_member_trait : no_connect_member
{
};

template <typename S, typename R>
struct connect_member_trait<S, R,
  void_t<
    decltype(declval<S>().connect(declval<R>()))
  >>
{
  static constexpr bool is_valid = true;

  using result_type = decltype(
    declval<S>().connect(declval<R>()));

  static constexpr bool is_noexcept =
    noexcept(declval<S>().connect(declval<R>()));
};

#else // defined(ASIO_HAS_DEDUCED_CONNECT_MEMBER_TRAIT)

template <typename S, typename R, typename = void>
struct connect_member_trait :
  conditional_t<
    is_same<S, remove_reference_t<S>>::value
      && is_same<R, decay_t<R>>::value,
    conditional_t<
      is_same<S, add_const_t<S>>::value,
      no_connect_member,
      traits::connect_member<add_const_t<S>, R>
    >,
    traits::connect_member<
      remove_reference_t<S>,
      decay_t<R>>
  >
{
};

#endif // defined(ASIO_HAS_DEDUCED_CONNECT_MEMBER_TRAIT)

} // namespace detail
namespace traits {

template <typename S, typename R, typename>
struct connect_member_default :
  detail::connect_member_trait<S, R>
{
};

template <typename S, typename R, typename>
struct connect_member :
  connect_member_default<S, R>
{
};

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TRAITS_CONNECT_MEMBER_HPP
