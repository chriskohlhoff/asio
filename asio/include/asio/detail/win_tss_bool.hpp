//
// win_tss_bool.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_TSS_BOOL_HPP
#define ASIO_DETAIL_WIN_TSS_BOOL_HPP

#include "asio/detail/push_options.hpp"

#if defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <new>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

class win_tss_bool
{
public:
  // Constructor.
  win_tss_bool()
  {
    tss_key_ = ::TlsAlloc();
    if (tss_key_ == TLS_OUT_OF_INDEXES)
      throw std::bad_alloc();
  }

  // Destructor.
  ~win_tss_bool()
  {
    ::TlsFree(tss_key_);
  }

  // Test the value of the flag.
  operator bool() const
  {
    return ::TlsGetValue(tss_key_) != 0;
  }

  // Test for the value of the flag being false.
  bool operator!() const
  {
    return ::TlsGetValue(tss_key_) == 0;
  }

  // Set the value of the flag.
  void operator=(bool value)
  {
    ::TlsSetValue(tss_key_, value ? this : 0);
  }

private:
  // Thread-specific storage to allow unlocked access to determine whether a
  // thread is a member of the pool.
  mutable unsigned long tss_key_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_TSS_BOOL_HPP
