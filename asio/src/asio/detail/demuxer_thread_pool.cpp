//
// demuxer_thread_pool.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/demuxer_thread_pool.hpp"
#include <boost/thread.hpp>

namespace asio {
namespace detail {

demuxer_thread_pool::
demuxer_thread_pool()
{
#if defined(_WIN32)
  tss_key_ = ::TlsAlloc();
  if (tss_key_ == TLS_OUT_OF_INDEXES)
    throw boost::thread_resource_error();
#else // defined(_WIN32)
  if (::pthread_key_create(&tss_key_, 0) != 0)
    throw boost::thread_resource_error();
#endif // defined(_WIN32)
}

demuxer_thread_pool::
~demuxer_thread_pool()
{
#if defined(_WIN32)
  ::TlsFree(tss_key_);
#else // defined(_WIN32)
  ::pthread_key_delete(tss_key_);
#endif // defined(_WIN32)
}

void
demuxer_thread_pool::
add_current_thread()
{
  set_current_thread_is_member(true);
}

void
demuxer_thread_pool::
remove_current_thread()
{
  set_current_thread_is_member(false);
}

bool
demuxer_thread_pool::
current_thread_is_member() const
{
#if defined(_WIN32)
  return ::TlsGetValue(tss_key_) != 0;
#else // defined(_WIN32)
  return ::pthread_getspecific(tss_key_) != 0;
#endif // defined(_WIN32)
}

void
demuxer_thread_pool::
set_current_thread_is_member(
  bool is_member)
{
#if defined(_WIN32)
  ::TlsSetValue(tss_key_, is_member ? this : 0);
#else // defined(_WIN32)
  ::pthread_setspecific(tss_key_, is_member ? this : 0);
#endif // defined(_WIN32)
}

} // namespace detail
} // namespace asio
