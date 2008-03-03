//
// posix_event.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <boost/assert.hpp>
#include <boost/throw_exception.hpp>
#include <pthread.h>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
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
    int error = ::pthread_cond_init(&cond_, 0);
    if (error != 0)
    {
      asio::system_error e(
          asio::error_code(error,
            asio::error::get_system_category()),
          "event");
      boost::throw_exception(e);
    }
  }

  // Destructor.
  ~posix_event()
  {
    ::pthread_cond_destroy(&cond_);
  }

  // Signal the event.
  template <typename Lock>
  void signal(Lock& lock)
  {
    BOOST_ASSERT(lock.locked());
    (void)lock;
    signalled_ = true;
    ::pthread_cond_signal(&cond_); // Ignore EINVAL.
  }

  // Reset the event.
  template <typename Lock>
  void clear(Lock& lock)
  {
    BOOST_ASSERT(lock.locked());
    (void)lock;
    signalled_ = false;
  }

  // Wait for the event to become signalled.
  template <typename Lock>
  void wait(Lock& lock)
  {
    BOOST_ASSERT(lock.locked());
    while (!signalled_)
      ::pthread_cond_wait(&cond_, &lock.mutex().mutex_); // Ignore EINVAL.
  }

private:
  ::pthread_cond_t cond_;
  bool signalled_;
};

} // namespace detail
} // namespace asio

#endif // defined(BOOST_HAS_PTHREADS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_POSIX_EVENT_HPP
