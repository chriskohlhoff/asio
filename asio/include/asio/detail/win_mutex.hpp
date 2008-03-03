//
// win_mutex.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_MUTEX_HPP
#define ASIO_DETAIL_WIN_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#if defined(BOOST_WINDOWS)

#include "asio/error.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/scoped_lock.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

class win_mutex
  : private noncopyable
{
public:
  typedef asio::detail::scoped_lock<win_mutex> scoped_lock;

  // Constructor.
  win_mutex()
  {
    int error = do_init();
    if (error != 0)
    {
      asio::system_error e(
          asio::error_code(error,
            asio::error::get_system_category()),
          "mutex");
      boost::throw_exception(e);
    }
  }

  // Destructor.
  ~win_mutex()
  {
    ::DeleteCriticalSection(&crit_section_);
  }

  // Lock the mutex.
  void lock()
  {
    int error = do_lock();
    if (error != 0)
    {
      asio::system_error e(
          asio::error_code(error,
            asio::error::get_system_category()),
          "mutex");
      boost::throw_exception(e);
    }
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
  int do_init()
  {
#if defined(__MINGW32__)
    // Not sure if MinGW supports structured exception handling, so for now
    // we'll just call the Windows API and hope.
    ::InitializeCriticalSection(&crit_section_);
    return 0;
#else
    __try
    {
      ::InitializeCriticalSection(&crit_section_);
    }
    __except(GetExceptionCode() == STATUS_NO_MEMORY
        ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
    {
      return ERROR_OUTOFMEMORY;
    }

    return 0;
#endif
  }

  // Locking must be performed in a separate function to lock() since the
  // compiler does not support the use of structured exceptions and C++
  // exceptions in the same function.
  int do_lock()
  {
#if defined(__MINGW32__)
    // Not sure if MinGW supports structured exception handling, so for now
    // we'll just call the Windows API and hope.
    ::EnterCriticalSection(&crit_section_);
    return 0;
#else
    __try
    {
      ::EnterCriticalSection(&crit_section_);
    }
    __except(GetExceptionCode() == STATUS_INVALID_HANDLE
        || GetExceptionCode() == STATUS_NO_MEMORY
        ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
    {
      if (GetExceptionCode() == STATUS_NO_MEMORY)
        return ERROR_OUTOFMEMORY;
      return ERROR_INVALID_HANDLE;
    }

    return 0;
#endif
  }

  ::CRITICAL_SECTION crit_section_;
};

} // namespace detail
} // namespace asio

#endif // defined(BOOST_WINDOWS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_MUTEX_HPP
