//
// win_mutex.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_MUTEX_HPP
#define ASIO_DETAIL_WIN_MUTEX_HPP

#include "asio/detail/push_options.hpp"

#if defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <new>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_types.hpp"
#include "asio/detail/scoped_lock.hpp"

namespace asio {
namespace detail {

class win_mutex
  : private boost::noncopyable
{
public:
  typedef asio::detail::scoped_lock<win_mutex> scoped_lock;

  // Constructor.
  win_mutex()
  {
    if (!do_init())
      throw std::bad_alloc();
  }

  // Destructor.
  ~win_mutex()
  {
    ::DeleteCriticalSection(&crit_section_);
  }

  // Lock the mutex.
  void lock()
  {
    if (!do_lock())
      throw std::runtime_error("Unable to lock mutex");
  }

  // Unlock the mutex.
  void unlock()
  {
    ::LeaveCriticalSection(&crit_section_);
  }

private:
  // Initialisation must be performed in a separate function to the constructor
  // since the compiler does not support the use of structured exceptions and
  // C++ exceptions in the same function.
  bool do_init()
  {
#if defined(__MINGW32__)
    // Not sure if MinGW supports structured exception handling, so for now
    // we'll just call the Windows API and hope.
    ::InitializeCriticalSection(&crit_section_);
    return true;
#else
    __try
    {
      ::InitializeCriticalSection(&crit_section_);
    }
    __except(GetExceptionCode() == STATUS_NO_MEMORY
        ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
    {
      return false;
    }

    return true;
#endif
  }

  // Locking must be performed in a separate function to lock() since the
  // compiler does not support the use of structured exceptions and C++
  // exceptions in the same function.
  bool do_lock()
  {
#if defined(__MINGW32__)
    // Not sure if MinGW supports structured exception handling, so for now
    // we'll just call the Windows API and hope.
    ::EnterCriticalSection(&crit_section_);
    return true;
#else
    __try
    {
      ::EnterCriticalSection(&crit_section_);
    }
    __except(GetExceptionCode() == STATUS_INVALID_HANDLE
        ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
    {
      return false;
    }

    return true;
#endif
  }

  ::CRITICAL_SECTION crit_section_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_MUTEX_HPP
