//
// posix_mutex.hpp
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

#ifndef ASIO_DETAIL_POSIX_MUTEX_HPP
#define ASIO_DETAIL_POSIX_MUTEX_HPP

#include "asio/detail/push_options.hpp"

#if !defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include <pthread.h>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/scoped_lock.hpp"

namespace asio {
namespace detail {

class posix_mutex
  : private boost::noncopyable
{
public:
  typedef asio::detail::scoped_lock<posix_mutex> scoped_lock;

  // Constructor.
  posix_mutex()
  {
    ::pthread_mutex_init(&mutex_, 0);
  }

  // Destructor.
  ~posix_mutex()
  {
    ::pthread_mutex_destroy(&mutex_);
  }

  // Lock the mutex.
  void lock()
  {
    ::pthread_mutex_lock(&mutex_);
  }

  // Unlock the mutex.
  void unlock()
  {
    ::pthread_mutex_unlock(&mutex_);
  }

private:
  ::pthread_mutex_t mutex_;
};

} // namespace detail
} // namespace asio

#endif // !defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_POSIX_MUTEX_HPP
