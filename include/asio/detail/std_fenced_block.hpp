//
// detail/std_fenced_block.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_STD_FENCED_BLOCK_HPP
#define ASIO_DETAIL_STD_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <atomic>
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

#if defined(__has_feature)
# if __has_feature(thread_sanitizer)
#  include "sanitizer/tsan_interface.h"
#  define USE_TSAN_SEMANTICS 1
# else
#  define USE_TSAN_SEMANTICS 0
# endif
#else
# define USE_TSAN_SEMANTICS 0
#endif

namespace asio {
ASIO_INLINE_NAMESPACE_BEGIN
namespace detail {

class std_fenced_block
  : private noncopyable
{
public:
  enum half_t { half };
  enum full_t { full };

  // Constructor for a half fenced block.
  explicit std_fenced_block(half_t)
  {
  }

  // Constructor for a full fenced block.
  explicit std_fenced_block(full_t)
  {
#if USE_TSAN_SEMANTICS
    __tsan_acquire(&tsan);
#else
    std::atomic_thread_fence(std::memory_order_acquire);
#endif
  }

  // Destructor.
  ~std_fenced_block()
  {
#if USE_TSAN_SEMANTICS
    __tsan_release(&tsan);
#else
    std::atomic_thread_fence(std::memory_order_release);
#endif
  }

#if USE_TSAN_SEMANTICS
private:
    int tsan;
#endif
};

} // namespace detail
ASIO_INLINE_NAMESPACE_END
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_STD_FENCED_BLOCK_HPP
