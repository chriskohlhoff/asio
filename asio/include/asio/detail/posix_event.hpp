//
// posix_event.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_POSIX_EVENT_HPP
#define ASIO_DETAIL_POSIX_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#if defined(BOOST_HAS_PTHREADS)

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include <pthread.h>
#include "asio/detail/pop_options.hpp"

#include "asio/system_error.hpp"
#include "asio/detail/noncopyable.hpp"

namespace asio {
namespace detail {

class posix_event
  : private noncopyable
{
public:
  // Constructor.
  posix_event()
    : signalled_(false)
  {
    int error = ::pthread_mutex_init(&mutex_, 0);
    if (error != 0)
    {
      asio::system_error e(
          asio::error_code(error, asio::native_ecat),
          "event");
      boost::throw_exception(e);
    }

    error = ::pthread_cond_init(&cond_, 0);
    if (error != 0)
    {
      ::pthread_mutex_destroy(&mutex_);
      asio::system_error e(
          asio::error_code(error, asio::native_ecat),
          "event");
      boost::throw_exception(e);
    }
  }

  // Destructor.
  ~posix_event()
  {
    ::pthread_cond_destroy(&cond_);
    ::pthread_mutex_destroy(&mutex_);
  }

  // Signal the event.
  void signal()
  {
    ::pthread_mutex_lock(&mutex_); // Ignore EINVAL and EDEADLK.
    signalled_ = true;
    ::pthread_cond_signal(&cond_); // Ignore EINVAL.
    ::pthread_mutex_unlock(&mutex_); // Ignore EINVAL and EPERM.
  }

  // Reset the event.
  void clear()
  {
    ::pthread_mutex_lock(&mutex_); // Ignore EINVAL and EDEADLK.
    signalled_ = false;
    ::pthread_mutex_unlock(&mutex_); // Ignore EINVAL and EPERM.
  }

  // Wait for the event to become signalled.
  void wait()
  {
    ::pthread_mutex_lock(&mutex_); // Ignore EINVAL and EDEADLK.
    while (!signalled_)
      ::pthread_cond_wait(&cond_, &mutex_); // Ignore EINVAL.
    ::pthread_mutex_unlock(&mutex_); // Ignore EINVAL and EPERM.
  }

private:
  ::pthread_mutex_t mutex_;
  ::pthread_cond_t cond_;
  bool signalled_;
};

} // namespace detail
} // namespace asio

#endif // defined(BOOST_HAS_PTHREADS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_POSIX_EVENT_HPP
