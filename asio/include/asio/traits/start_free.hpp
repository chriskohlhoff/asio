//
// traits/start_free.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TRAITS_START_FREE_HPP
#define ASIO_TRAITS_START_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_START_FREE_TRAIT 1
#endif // defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename T, typename = void>
struct start_free_default;

template <typename T, typename = void>
struct start_free;

} // namespace traits
namespace detail {

struct no_start_free
{
  static constexpr bool is_valid = false;
  static constexpr bool is_noexcept = false;
};

#if defined(ASIO_HAS_DEDUCED_START_FREE_TRAIT)

template <typename T, typename = void>
struct start_free_trait : no_start_free
{
};

template <typename T>
struct start_free_trait<T,
  void_t<
    decltype(start(declval<T>()))
  >>
{
  static constexpr bool is_valid = true;

  using result_type = decltype(start(declval<T>()));

  static constexpr bool is_noexcept = noexcept(start(declval<T>()));
};

#else // defined(ASIO_HAS_DEDUCED_START_FREE_TRAIT)

template <typename T, typename = void>
struct start_free_trait :
  conditional_t<
    is_same<T, remove_reference_t<T>>::value,
    conditional_t<
      is_same<T, add_const_t<T>>::value,
      no_start_free,
      traits::start_free<add_const_t<T>>
    >,
    traits::start_free<remove_reference_t<T>>
  >
{
};

#endif // defined(ASIO_HAS_DEDUCED_START_FREE_TRAIT)

} // namespace detail
namespace traits {

template <typename T, typename>
struct start_free_default :
  detail::start_free_trait<T>
{
};

template <typename T, typename>
struct start_free :
  start_free_default<T>
{
};

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TRAITS_START_FREE_HPP
