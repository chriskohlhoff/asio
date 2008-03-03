//
// posix_mutex.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_POSIX_MUTEX_HPP
#define ASIO_DETAIL_POSIX_MUTEX_HPP

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

#include "asio/error.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/scoped_lock.hpp"

namespace asio {
namespace detail {

class posix_event;

class posix_mutex
  : private noncopyable
{
public:
  typedef asio::detail::scoped_lock<posix_mutex> scoped_lock;

  // Constructor.
  posix_mutex()
  {
    int error = ::pthread_mutex_init(&mutex_, 0);
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
  ~posix_mutex()
  {
    ::pthread_mutex_destroy(&mutex_);
  }

  // Lock the mutex.
  void lock()
  {
    int error = ::pthread_mutex_lock(&mutex_);
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
    int error = ::pthread_mutex_unlock(&mutex_);
    if (error != 0)
    {
      asio::system_error e(
          asio::error_code(error,
            asio::error::get_system_category()),
          "mutex");
      boost::throw_exception(e);
    }
  }

private:
  friend class posix_event;
  ::pthread_mutex_t mutex_;
};

} // namespace detail
} // namespace asio

#endif // defined(BOOST_HAS_PTHREADS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_POSIX_MUTEX_HPP
