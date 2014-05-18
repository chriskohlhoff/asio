//
// unspecified_allocator.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_UNSPECIFIED_ALLOCATOR_HPP
#define ASIO_UNSPECIFIED_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/memory.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// An allocator for functions that do not explicitly specify one.
/**
 * The unspecified allocator is used for function objects that do not explicitly
 * specify an associated allocator. It is implemented in terms of
 * @c std::allocator.
 */
template <typename T>
class unspecified_allocator
  : public std::allocator<T>
{
public:
  /// Default constructor.
  unspecified_allocator() ASIO_NOEXCEPT
  {
  }

  /// Copy constructor.
  unspecified_allocator(const unspecified_allocator& other) ASIO_NOEXCEPT
    : std::allocator<T>(static_cast<const std::allocator<T>&>(other))
  {
  }

  /// Construct from an @c unspecified_allocator for another type.
  template <typename U>
  unspecified_allocator(
      const unspecified_allocator<U>& other) ASIO_NOEXCEPT
    : std::allocator<T>(static_cast<const std::allocator<U>&>(other))
  {
  }

  /// Determine @c unspecified_allocator to use for another type.
  template <typename U>
  struct rebind
  {
    typedef unspecified_allocator<U> other;
  };
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_UNSPECIFIED_ALLOCATOR_HPP
