//
// detail/impl/posix_mutex.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_POSIX_MUTEX_HPP
#define ASIO_DETAIL_IMPL_POSIX_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#if defined(BOOST_HAS_PTHREADS) && !defined(ASIO_DISABLE_THREADS)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

inline posix_mutex::~posix_mutex()
{
  ::pthread_mutex_destroy(&mutex_); // Ignore EBUSY.
}

inline void posix_mutex::lock()
{
  (void)::pthread_mutex_lock(&mutex_); // Ignore EINVAL.
}

// Unlock the mutex.
inline void posix_mutex::unlock()
{
  (void)::pthread_mutex_unlock(&mutex_); // Ignore EINVAL.
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(BOOST_HAS_PTHREADS) && !defined(ASIO_DISABLE_THREADS)

#endif // ASIO_DETAIL_IMPL_POSIX_MUTEX_HPP
