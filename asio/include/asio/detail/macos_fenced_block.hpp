//
// detail/macos_fenced_block.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2016 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_MACOS_FENCED_BLOCK_HPP
#define ASIO_DETAIL_MACOS_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(__MACH__) && defined(__APPLE__)

#include <AvailabilityMacros.h>

// Availability.h was introduced for supporting Mac OSX 10.6 and iOS
#if MAC_OS_X_VERSION_MAX_ALLOWED >= 1060 \
  || defined(__IPHONE_OS_VERSION_MIN_REQUIRED)
#include <Availability.h>
#endif

#if defined(__IPHONE_OS_VERSION_MIN_REQUIRED)

// OSMemoryBarrier was deprecated in iOS 10
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 100000
#define ASIO_APPLE_USE_ATOMIC_FENCE 1
#endif

// Use MAC_OS_X_VERSION_MIN_REQUIRED instead of __MAC_OS_X_VERSION_MIN_REQUIRED
// to keep compatibility with AvailabilityMacros.h
#elif defined(MAC_OS_X_VERSION_MIN_REQUIRED)

// OSMemoryBarrier was deprecated in Mac OSX 10.12
#if MAC_OS_X_VERSION_MAX_ALLOWED >= 101200
#define ASIO_APPLE_USE_ATOMIC_FENCE 1
#endif

#else
#error "Unknown Apple OS"
#endif

#ifdef ASIO_APPLE_USE_ATOMIC_FENCE
#include <atomic>
#else
#include <libkern/OSAtomic.h>
#endif

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class macos_fenced_block
  : private noncopyable
{
public:
  enum half_t { half };
  enum full_t { full };

  // Constructor for a half fenced block.
  explicit macos_fenced_block(half_t)
  {
  }

  // Constructor for a full fenced block.
  explicit macos_fenced_block(full_t)
  {
#ifdef ASIO_APPLE_USE_ATOMIC_FENCE
    std::atomic_thread_fence(std::memory_order_acquire);
#else
    OSMemoryBarrier();
#endif
  }

  // Destructor.
  ~macos_fenced_block()
  {
#ifdef ASIO_APPLE_USE_ATOMIC_FENCE
    std::atomic_thread_fence(std::memory_order_release);
#else
    OSMemoryBarrier();
#endif
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(__MACH__) && defined(__APPLE__)

#endif // ASIO_DETAIL_MACOS_FENCED_BLOCK_HPP
