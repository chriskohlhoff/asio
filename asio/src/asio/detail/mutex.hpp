//
// mutex.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_MUTEX_HPP
#define ASIO_DETAIL_MUTEX_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <stdexcept>
#include <boost/noncopyable.hpp>
#if defined(_WIN32)
#include "asio/detail/socket_types.hpp"
#else
#include <pthread.h>
#endif // defined(_WIN32)
#include "asio/detail/pop_options.hpp"

#include "asio/detail/scoped_lock.hpp"

namespace asio {
namespace detail {

class mutex
  : private boost::noncopyable
{
public:
  typedef asio::detail::scoped_lock<mutex> scoped_lock;

  // Constructor.
  mutex()
  {
#if defined(_WIN32)
    ::InitializeCriticalSection(&crit_section_);
#else // defined(_WIN32)
#endif // defined(_WIN32)
  }

  // Destructor.
  ~mutex()
  {
#if defined(_WIN32)
    ::DeleteCriticalSection(&crit_section_);
#else // defined(_WIN32)
#endif // defined(_WIN32)
  }

  // Lock the mutex.
  void lock()
  {
#if defined(_WIN32)
    ::EnterCriticalSection(&crit_section_);
#else // defined(_WIN32)
#endif // defined(_WIN32)
  }

  // Unlock the mutex.
  void unlock()
  {
#if defined(_WIN32)
    ::LeaveCriticalSection(&crit_section_);
#else // defined(_WIN32)
#endif // defined(_WIN32)
  }

private:
#if defined(_WIN32)
  CRITICAL_SECTION crit_section_;
#else // defined(_WIN32)
#endif // defined(_WIN32)
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_MUTEX_HPP
