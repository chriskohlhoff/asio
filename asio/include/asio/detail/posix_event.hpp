//
// posix_event.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
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

#if !defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include <pthread.h>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

class posix_event
  : private boost::noncopyable
{
public:
  // Constructor.
  posix_event()
    : signalled_(false)
  {
    ::pthread_mutex_init(&mutex_, 0);
    ::pthread_cond_init(&cond_, 0);
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
    ::pthread_mutex_lock(&mutex_);
    signalled_ = true;
    ::pthread_cond_signal(&cond_);
    ::pthread_mutex_unlock(&mutex_);
  }

  // Reset the event.
  void clear()
  {
    ::pthread_mutex_lock(&mutex_);
    signalled_ = false;
    ::pthread_mutex_unlock(&mutex_);
  }

  // Wait for the event to become signalled.
  void wait()
  {
    ::pthread_mutex_lock(&mutex_);
    while (!signalled_)
      ::pthread_cond_wait(&cond_, &mutex_);
    ::pthread_mutex_unlock(&mutex_);
  }

private:
  ::pthread_mutex_t mutex_;
  ::pthread_cond_t cond_;
  bool signalled_;
};

} // namespace detail
} // namespace asio

#endif // !defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_POSIX_EVENT_HPP
