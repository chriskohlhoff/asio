//
// tss_bool.hpp
// ~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_TSS_BOOL_HPP
#define ASIO_DETAIL_TSS_BOOL_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <stdexcept>
#include <boost/noncopyable.hpp>
#if defined(_WIN32)
#include "asio/detail/socket_types.hpp"
#else
#include <pthread.h>
#endif // defined(_WIN32)
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

class tss_bool
{
public:
  // Constructor.
  tss_bool()
  {
#if defined(_WIN32)
    tss_key_ = ::TlsAlloc();
    if (tss_key_ == TLS_OUT_OF_INDEXES)
      throw std::runtime_error("Cannot create thread-local storage");
#else // defined(_WIN32)
    if (::pthread_key_create(&tss_key_, 0) != 0)
      throw std::runtime_error("Cannot create thread-local storage");
#endif // defined(_WIN32)
  }

  // Destructor.
  ~tss_bool()
  {
#if defined(_WIN32)
    ::TlsFree(tss_key_);
#else // defined(_WIN32)
    ::pthread_key_delete(tss_key_);
#endif // defined(_WIN32)
  }

  // Test the value of the flag.
  operator bool() const
  {
#if defined(_WIN32)
    return ::TlsGetValue(tss_key_) != 0;
#else // defined(_WIN32)
    return ::pthread_getspecific(tss_key_) != 0;
#endif // defined(_WIN32)
  }

  // Test for the value of the flag being false.
  bool operator!() const
  {
#if defined(_WIN32)
    return ::TlsGetValue(tss_key_) == 0;
#else // defined(_WIN32)
    return ::pthread_getspecific(tss_key_) == 0;
#endif // defined(_WIN32)
  }

  // Set the value of the flag.
  void operator=(bool value)
  {
#if defined(_WIN32)
    ::TlsSetValue(tss_key_, value ? this : 0);
#else // defined(_WIN32)
    ::pthread_setspecific(tss_key_, value ? this : 0);
#endif // defined(_WIN32)
  }

private:
  // Thread-specific storage to allow unlocked access to determine whether a
  // thread is a member of the pool.
#if defined(_WIN32)
  mutable unsigned long tss_key_;
#else
  mutable pthread_key_t tss_key_;
#endif
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TSS_BOOL_HPP
