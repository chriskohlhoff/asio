//
// posix_event.hpp
// ~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_POSIX_EVENT_HPP
#define ASIO_DETAIL_POSIX_EVENT_HPP

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
