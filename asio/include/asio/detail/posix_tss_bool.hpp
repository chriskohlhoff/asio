//
// posix_tss_bool.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_POSIX_TSS_BOOL_HPP
#define ASIO_DETAIL_POSIX_TSS_BOOL_HPP

#include "asio/detail/push_options.hpp"

#if !defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <stdexcept>
#include <pthread.h>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

class posix_tss_bool
{
public:
  // Constructor.
  posix_tss_bool()
  {
    if (::pthread_key_create(&tss_key_, 0) != 0)
      throw std::runtime_error("Cannot create thread-local storage");
  }

  // Destructor.
  ~posix_tss_bool()
  {
    ::pthread_key_delete(tss_key_);
  }

  // Test the value of the flag.
  operator bool() const
  {
    return ::pthread_getspecific(tss_key_) != 0;
  }

  // Test for the value of the flag being false.
  bool operator!() const
  {
    return ::pthread_getspecific(tss_key_) == 0;
  }

  // Set the value of the flag.
  void operator=(bool value)
  {
    ::pthread_setspecific(tss_key_, value ? this : 0);
  }

private:
  // Thread-specific storage to allow unlocked access to determine whether a
  // thread is a member of the pool.
  mutable pthread_key_t tss_key_;
};

} // namespace detail
} // namespace asio

#endif // !defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_POSIX_TSS_BOOL_HPP
