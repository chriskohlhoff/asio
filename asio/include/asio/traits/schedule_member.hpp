//
// traits/schedule_member.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TRAITS_SCHEDULE_MEMBER_HPP
#define ASIO_TRAITS_SCHEDULE_MEMBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_SCHEDULE_MEMBER_TRAIT 1
#endif // defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename T, typename = void>
struct schedule_member_default;

template <typename T, typename = void>
struct schedule_member;

} // namespace traits
namespace detail {

struct no_schedule_member
{
  static constexpr bool is_valid = false;
  static constexpr bool is_noexcept = false;
};

#if defined(ASIO_HAS_DEDUCED_SCHEDULE_MEMBER_TRAIT)

template <typename T, typename = void>
struct schedule_member_trait : no_schedule_member
{
};

template <typename T>
struct schedule_member_trait<T,
  void_t<
    decltype(declval<T>().schedule())
  >>
{
  static constexpr bool is_valid = true;

  using result_type = decltype(declval<T>().schedule());

  static constexpr bool is_noexcept = noexcept(declval<T>().schedule());
};

#else // defined(ASIO_HAS_DEDUCED_SCHEDULE_MEMBER_TRAIT)

template <typename T, typename = void>
struct schedule_member_trait :
  conditional_t<
    is_same<T, remove_reference_t<T>>::value,
    conditional_t<
      is_same<T, add_const_t<T>>::value,
      no_schedule_member,
      traits::schedule_member<add_const_t<T>>
    >,
    traits::schedule_member<remove_reference_t<T>>
  >
{
};

#endif // defined(ASIO_HAS_DEDUCED_SCHEDULE_MEMBER_TRAIT)

} // namespace detail
namespace traits {

template <typename T, typename>
struct schedule_member_default :
  detail::schedule_member_trait<T>
{
};

template <typename T, typename>
struct schedule_member :
  schedule_member_default<T>
{
};

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TRAITS_SCHEDULE_MEMBER_HPP
