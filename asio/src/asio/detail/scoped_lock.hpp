//
// scoped_lock.hpp
// ~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SCOPED_LOCK_HPP
#define ASIO_DETAIL_SCOPED_LOCK_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

// Helper class to lock and unlock a mutex automatically.
template <typename Mutex>
class scoped_lock
  : private boost::noncopyable
{
public:
  // Constructor acquires the lock.
  scoped_lock(Mutex& m)
    : mutex_(m)
  {
    mutex_.lock();
    locked_ = true;
  }

  // Destructor releases the lock.
  ~scoped_lock()
  {
    if (locked_)
      mutex_.unlock();
  }

  // Explicitly acquire the lock.
  void lock()
  {
    if (!locked_)
    {
      mutex_.lock();
      locked_ = true;
    }
  }

  // Explicitly release the lock.
  void unlock()
  {
    if (locked_)
    {
      mutex_.unlock();
      locked_ = false;
    }
  }

private:
  // The underlying mutex.
  Mutex& mutex_;

  // Whether the mutex is currently locked or unlocked.
  bool locked_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SCOPED_LOCK_HPP
