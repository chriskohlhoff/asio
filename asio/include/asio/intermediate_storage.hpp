//
// intermediate_storage.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_INTERMEDIATE_STORAGE_HPP
#define ASIO_INTERMEDIATE_STORAGE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename>
struct intermediate_storage_check
{
  typedef void type;
};

template <typename T, typename Args, typename = void>
struct intermediate_storage_impl
{
  typedef void type;
};

template <typename T, typename... Args>
struct intermediate_storage_impl<T, void(Args...),
  typename intermediate_storage_check<
    typename T::template intermediate_storage<Args...>::type>::type>
{
  typedef typename T::template intermediate_storage<Args...>::type type;
};

} // namespace detail

#if defined(GENERATING_DOCUMENTATION)

/// Traits type used to determine the storage requirements of an operation.
/**
 * A program may specialise this traits type if the @c T template parameter in
 * the specialisation is a user-defined type.
 *
 * Specialisations of this trait may provide a nested typedef @c type, which is
 * a trivial standard-layout type suitable for use as uninitialised storage by
 * the operation initiated by the type @c T. If the operation has no fixed-size
 * storage requirement, this type is @c void.
 */
template <typename T, typename... Args>
struct intermediate_storage
{
  /// If @c T has a nested class template @c intermediate_storage such that
  /// <tt>T::template intermediate_storage<Args...>::type</tt> is a valid type,
  /// <tt>T::intermediate_storage<Args...>::type</tt. Otherwise the typedef
  /// @c type is @c void.
  typedef see_below type;
};
#else
template <typename T, typename... Args>
struct intermediate_storage
  : detail::intermediate_storage_impl<T, void(Args...)>
{
};
#endif

#if defined(ASIO_HAS_ALIAS_TEMPLATES)

template <typename T, typename... Args>
using intermediate_storage_t = typename intermediate_storage<T, Args...>::type;

#endif // defined(ASIO_HAS_ALIAS_TEMPLATES)

/// Determine the appropriate intermediate storage type as a union of types.
/**
 * This helper template automatically determines the correct intermediate
 * storage type as a union of two other types. If either or both of those
 * types are void, then the "union" type is also void.
 */
template <typename T, typename U>
struct intermediate_storage_union
{
#if defined(GENERATING_DOCUMENTATION)
  /// If either of T or U are void, void. Otherwise a suitable union storage
  /// type.
  typedef see_below type;
#else // defined(GENERATING_DOCUMENTATION)
  union type
  {
    T t;
    U u;
  };
#endif // defined(GENERATING_DOCUMENTATION)
};

#if !defined(GENERATING_DOCUMENTATION)
template <typename T>
struct intermediate_storage_union<T, void>
{
  typedef void type;
};

template <typename U>
struct intermediate_storage_union<void, U>
{
  typedef void type;
};

template <>
struct intermediate_storage_union<void, void>
{
  typedef void type;
};
#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_INTERMEDIATE_STORAGE_HPP
