//
// traits/submit_member.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TRAITS_SUBMIT_MEMBER_HPP
#define ASIO_TRAITS_SUBMIT_MEMBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_SUBMIT_MEMBER_TRAIT 1
#endif // defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename S, typename R, typename = void>
struct submit_member_default;

template <typename S, typename R, typename = void>
struct submit_member;

} // namespace traits
namespace detail {

struct no_submit_member
{
  static constexpr bool is_valid = false;
  static constexpr bool is_noexcept = false;
};

#if defined(ASIO_HAS_DEDUCED_SUBMIT_MEMBER_TRAIT)

template <typename S, typename R, typename = void>
struct submit_member_trait : no_submit_member
{
};

template <typename S, typename R>
struct submit_member_trait<S, R,
  void_t<
    decltype(declval<S>().submit(declval<R>()))
  >>
{
  static constexpr bool is_valid = true;

  using result_type = decltype(
    declval<S>().submit(declval<R>()));

  static constexpr bool is_noexcept =
    noexcept(declval<S>().submit(declval<R>()));
};

#else // defined(ASIO_HAS_DEDUCED_SUBMIT_MEMBER_TRAIT)

template <typename S, typename R, typename = void>
struct submit_member_trait :
  conditional_t<
    is_same<S, remove_reference_t<S>>::value
      && is_same<R, decay_t<R>>::value,
    conditional_t<
      is_same<S, add_const_t<S>>::value,
      no_submit_member,
      traits::submit_member<add_const_t<S>, R>
    >,
    traits::submit_member<
      remove_reference_t<S>,
      decay_t<R>>
  >
{
};

#endif // defined(ASIO_HAS_DEDUCED_SUBMIT_MEMBER_TRAIT)

} // namespace detail
namespace traits {

template <typename S, typename R, typename>
struct submit_member_default :
  detail::submit_member_trait<S, R>
{
};

template <typename S, typename R, typename>
struct submit_member :
  submit_member_default<S, R>
{
};

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TRAITS_SUBMIT_MEMBER_HPP
