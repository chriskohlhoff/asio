//
// counting_completion_context.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/counting_completion_context.hpp"

#include "asio/detail/push_options.hpp"
#include <cassert>
#include <queue>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/mutex.hpp"

namespace asio {

struct counting_completion_context::impl
{
public:
  // Constructor.
  impl(int max_concurrent_upcalls);

  // Mutex to protect access to internal data.
  detail::mutex mutex_;

  // The maximum number of concurrent upcalls.
  int max_concurrent_upcalls_;

  // The current number of upcalls.
  int concurrent_upcalls_;

  // Structure containing information about a locker.
  struct locker_info
  {
    completion_context_locker* locker;
    void* arg;
  };

  // The type of a list of lockers waiting for the completion context.
  typedef std::queue<locker_info> locker_queue;
  
  // The tasks that are waiting on the completion context.
  locker_queue waiting_lockers_;
};

counting_completion_context::impl::
impl(
    int max_concurrent_upcalls)
  : mutex_(),
    max_concurrent_upcalls_(max_concurrent_upcalls),
    concurrent_upcalls_(0),
    waiting_lockers_()
{
}

counting_completion_context::
counting_completion_context(
    int max_concurrent_upcalls)
  : impl_(new impl(max_concurrent_upcalls))
{
}

counting_completion_context::
~counting_completion_context()
{
  delete impl_;
}

bool
counting_completion_context::
try_acquire()
{
  detail::mutex::scoped_lock lock(impl_->mutex_);

  assert(impl_->concurrent_upcalls_ <= impl_->max_concurrent_upcalls_);
  assert(impl_->concurrent_upcalls_ >= 0);

  if (impl_->concurrent_upcalls_ < impl_->max_concurrent_upcalls_)
  {
    ++impl_->concurrent_upcalls_;
    return true;
  }

  return false;
}

void
counting_completion_context::
acquire(
    completion_context_locker& locker,
    void* arg)
{
  detail::mutex::scoped_lock lock(impl_->mutex_);

  assert(impl_->concurrent_upcalls_ <= impl_->max_concurrent_upcalls_);
  assert(impl_->concurrent_upcalls_ >= 0);

  if (impl_->concurrent_upcalls_ < impl_->max_concurrent_upcalls_)
  {
    // The context can been acquired for the locker.
    ++impl_->concurrent_upcalls_;
    lock.unlock();
    notify_locker(locker, arg);
  }
  else
  {
    // Can't acquire the context for the locker right now, so put it on the
    // list of waiters.
    impl::locker_info info = { &locker, arg };
    impl_->waiting_lockers_.push(info);
  }
}

void
counting_completion_context::
release()
{
  detail::mutex::scoped_lock lock(impl_->mutex_);

  assert(impl_->concurrent_upcalls_ <= impl_->max_concurrent_upcalls_);
  assert(impl_->concurrent_upcalls_ > 0);

  // Check if we can start one of the waiting tasks now.
  if (impl_->concurrent_upcalls_ <= impl_->max_concurrent_upcalls_
      && !impl_->waiting_lockers_.empty())
  {
    impl::locker_info info = impl_->waiting_lockers_.front();
    impl_->waiting_lockers_.pop();
    lock.unlock();
    notify_locker(*info.locker, info.arg);
  }
  else
  {
    --impl_->concurrent_upcalls_;
  }
}

} // namespace asio
