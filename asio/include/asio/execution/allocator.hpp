//
// execution/allocator.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_ALLOCATOR_HPP
#define ASIO_EXECUTION_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

template <typename ProtoAllocator>
struct allocator_t
{
  /// The @c allocator property may be used with asio::execution::require.
  static constexpr bool is_requirable = true;

  /// The @c allocator property may be used with asio::execution::prefer.
  static constexpr bool is_preferable = true;

  /// Statically determines whether an Executor type provides the @c allocator
  /// property.
  template <typename Executor, typename Property = allocator_t,
    typename T = typename enable_if<
      (static_cast<void>(Executor::query(*static_cast<Property*>(0))), true),
      decltype(Executor::query(*static_cast<Property*>(0)))
    >::type>
  static constexpr T static_query_v = Executor::query(Property());

  /// Get the current value associated with the property object.
  constexpr ProtoAllocator value() const
  {
    return a_;
  }

private:
  friend class allocator_t<void>;

  explicit constexpr allocator_t(const ProtoAllocator& a)
    : a_(a)
  {
  }

  ProtoAllocator a_;
};

template <>
struct allocator_t<void>
{
  /// The @c allocator property may be used with asio::execution::require.
  static constexpr bool is_requirable = true;

  /// The @c allocator property may be used with asio::execution::prefer.
  static constexpr bool is_preferable = true;

  /// Statically determines whether an Executor type provides the @c allocator
  /// property.
  template <typename Executor, typename Property = allocator_t,
    typename T = typename enable_if<
      (static_cast<void>(Executor::query(*static_cast<Property*>(0))), true),
      decltype(Executor::query(*static_cast<Property*>(0)))
    >::type>
  static constexpr T static_query_v = Executor::query(Property());

  /// Create an @c allocator property object for a specific allocator type.
  template <typename OtherProtoAllocator>
  constexpr allocator_t<OtherProtoAllocator> operator()(
      const OtherProtoAllocator& a) const
  {
    return allocator_t<OtherProtoAllocator>(a);
  }
};

constexpr allocator_t<void> allocator;

template <typename ProtoAllocator>
template <typename Executor, typename Property, typename T>
constexpr T allocator_t<ProtoAllocator>::static_query_v;

template <typename Executor, typename Property, typename T>
constexpr T allocator_t<void>::static_query_v;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_ALLOCATOR_HPP
