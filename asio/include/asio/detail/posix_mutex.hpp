//
// posix_mutex.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

#if !defined(BOOST_WINDOWS)

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

#endif // !defined(BOOST_WINDOWS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_POSIX_MUTEX_HPP
