//
// posix_tss_bool.hpp
// ~~~~~~~~~~~~~~~~~~
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
