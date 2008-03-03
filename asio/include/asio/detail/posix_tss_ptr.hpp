//
// posix_tss_ptr.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_POSIX_TSS_PTR_HPP
#define ASIO_DETAIL_POSIX_TSS_PTR_HPP

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

namespace asio {
namespace detail {

template <typename T>
class posix_tss_ptr
  : private noncopyable
{
public:
  // Constructor.
  posix_tss_ptr()
  {
    int error = ::pthread_key_create(&tss_key_, 0);
    if (error != 0)
    {
      asio::system_error e(
          asio::error_code(error,
            asio::error::get_system_category()),
          "tss");
      boost::throw_exception(e);
    }
  }

  // Destructor.
  ~posix_tss_ptr()
  {
    ::pthread_key_delete(tss_key_);
  }

  // Get the value.
  operator T*() const
  {
    return static_cast<T*>(::pthread_getspecific(tss_key_));
  }

  // Set the value.
  void operator=(T* value)
  {
    ::pthread_setspecific(tss_key_, value);
  }

private:
  // Thread-specific storage to allow unlocked access to determine whether a
  // thread is a member of the pool.
  pthread_key_t tss_key_;
};

} // namespace detail
} // namespace asio

#endif // defined(BOOST_HAS_PTHREADS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_POSIX_TSS_PTR_HPP
