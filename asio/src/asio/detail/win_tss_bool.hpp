//
// win_tss_bool.hpp
// ~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_WIN_TSS_BOOL_HPP
#define ASIO_DETAIL_WIN_TSS_BOOL_HPP

#include "asio/detail/push_options.hpp"

#if defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <stdexcept>
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
      throw std::runtime_error("Cannot create thread-local storage");
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
