//
// detail/win_static_mutex.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2024 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_STATIC_MUTEX_HPP
#define ASIO_DETAIL_WIN_STATIC_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS)

#include "asio/detail/scoped_lock.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct win_static_mutex
{
  typedef asio::detail::scoped_lock<win_static_mutex> scoped_lock;

#if _WIN32_WINNT >= 0x0600
  // Initialise the mutex.
  ASIO_DECL void init() {}
#else // _WIN32_WINNT >= 0x0600
  // Initialise the mutex.
  ASIO_DECL void init();

  // Initialisation must be performed in a separate function to the "public"
  // init() function since the compiler does not support the use of structured
  // exceptions and C++ exceptions in the same function.
  ASIO_DECL int do_init();
#endif // _WIN32_WINNT >= 0x0600

  // Lock the mutex.
  void lock()
  {
#if _WIN32_WINNT >= 0x0600
    ::AcquireSRWLockExclusive(&srwlock_);
#else // _WIN32_WINNT >= 0x0600
    ::EnterCriticalSection(&crit_section_);
#endif // _WIN32_WINNT >= 0x0600
  }

  // Unlock the mutex.
  void unlock()
  {
#if _WIN32_WINNT >= 0x0600
    ::ReleaseSRWLockExclusive(&srwlock_);
#else // _WIN32_WINNT >= 0x0600
    ::LeaveCriticalSection(&crit_section_);
#endif // _WIN32_WINNT >= 0x0600
  }

#if _WIN32_WINNT >= 0x0600
  ::SRWLOCK srwlock_;
#else // _WIN32_WINNT >= 0x0600
  bool initialised_;
  ::CRITICAL_SECTION crit_section_;
#endif // _WIN32_WINNT >= 0x0600
};

#if _WIN32_WINNT >= 0x0600
# define ASIO_WIN_STATIC_MUTEX_INIT { SRWLOCK_INIT }
#else // _WIN32_WINNT >= 0x0600
#if defined(UNDER_CE)
# define ASIO_WIN_STATIC_MUTEX_INIT { false, { 0, 0, 0, 0, 0 } }
#else // defined(UNDER_CE)
# define ASIO_WIN_STATIC_MUTEX_INIT { false, { 0, 0, 0, 0, 0, 0 } }
#endif // defined(UNDER_CE)
#endif // _WIN32_WINNT >= 0x0600

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_static_mutex.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_WINDOWS)

#endif // ASIO_DETAIL_WIN_STATIC_MUTEX_HPP
