//
// demuxer_thread_pool.hpp
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

#ifndef ASIO_DETAIL_DEMUXER_THREAD_POOL_HPP
#define ASIO_DETAIL_DEMUXER_THREAD_POOL_HPP

#if defined(_WIN32)
#include "asio/detail/socket_types.hpp"
#else
#include <pthread.h>
#endif // defined(_WIN32)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class demuxer_thread_pool
{
public:
  // Constructor.
  demuxer_thread_pool();

  // Destructor.
  ~demuxer_thread_pool();

  // Add the current thread to the pool.
  void add_current_thread();

  // Remove the current thread from the pool.
  void remove_current_thread();

  // Returns true if the current thread is a member of the pool.
  bool current_thread_is_member() const;

private:
  // Helper function to set whether the current thread is a member.
  void set_current_thread_is_member(bool is_member);

  // Thread-specific storage to allow unlocked access to determine whether a
  // thread is a member of the pool.
#if defined(_WIN32)
  unsigned long tss_key_;
#else
  pthread_key_t tss_key_;
#endif
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_DEMUXER_THREAD_POOL_HPP
