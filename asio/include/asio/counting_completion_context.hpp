//
// counting_completion_context.hpp
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

#ifndef ASIO_COUNTING_COMPLETION_CONTEXT_HPP
#define ASIO_COUNTING_COMPLETION_CONTEXT_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/mutex.hpp"

namespace asio {

/// The counting_completion_context class is a concrete implementation of the
/// Completion_Context concept. It allows a limitation on the number of
/// concurrent upcalls to completion handlers that may be associated with the
/// context.
class counting_completion_context
{
public:
  /// Constructor.
  explicit counting_completion_context(int max_concurrent_upcalls)
    : mutex_(),
      max_concurrent_upcalls_(max_concurrent_upcalls),
      concurrent_upcalls_(0),
      first_waiter_(0),
      last_waiter_(0)
  {
  }

  /// Destructor.
  ~counting_completion_context()
  {
    while (first_waiter_)
    {
      waiter_base* delete_waiter = first_waiter_;
      first_waiter_ = first_waiter_->next_;
      delete delete_waiter;
    }
  }

  /// Attempt to acquire the right to make an upcall.
  bool try_acquire()
  {
    detail::mutex::scoped_lock lock(mutex_);

    if (concurrent_upcalls_ < max_concurrent_upcalls_)
    {
      ++concurrent_upcalls_;
      return true;
    }
    return false;
  }

  /// Acquire the right to make an upcall.
  template <typename Handler>
  void acquire(Handler handler)
  {
    detail::mutex::scoped_lock lock(mutex_);

    if (concurrent_upcalls_ < max_concurrent_upcalls_)
    {
      // The context can been acquired for the locker.
      ++concurrent_upcalls_;
      lock.unlock();
      handler();
    }
    else
    {
      // Can't acquire the context for the locker right now, so put it on the
      // list of waiters.
      waiter_base* new_waiter = new waiter<Handler>(handler);
      if (first_waiter_ == 0)
      {
        first_waiter_ = new_waiter;
        last_waiter_ = new_waiter;
      }
      else
      {
        last_waiter_->next_ = new_waiter;
        last_waiter_ = new_waiter;
      }
    }
  }

  /// Relinquish a previously granted right to make an upcall.
  void release()
  {
    detail::mutex::scoped_lock lock(mutex_);

    // Check if we can start one of the waiting tasks now.
    if (concurrent_upcalls_ <= max_concurrent_upcalls_ && first_waiter_)
    {
      waiter_base* next_waiter = first_waiter_;
      first_waiter_ = first_waiter_->next_;
      if (first_waiter_ == 0)
        last_waiter_ = 0;
      lock.unlock();
      next_waiter->notify();
      delete next_waiter;
    }
    else
    {
      --concurrent_upcalls_;
    }
  }

private:
  // Mutex to protect access to internal data.
  detail::mutex mutex_;

  // The maximum number of concurrent upcalls.
  int max_concurrent_upcalls_;

  // The current number of upcalls.
  int concurrent_upcalls_;

  // Base class for all waiter types.
  class waiter_base
  {
  public:
    waiter_base()
      : next_(0)
    {
    }

    virtual ~waiter_base()
    {
    }

    virtual void notify() = 0;

    waiter_base* next_;
  };

  // Class template for a waiter.
  template <typename Handler>
  class waiter
    : public waiter_base
  {
  public:
    waiter(Handler handler)
      : handler_(handler)
    {
    }

    virtual void notify()
    {
      handler_();
    }

  private:
    Handler handler_;
  };

  // The start of the list of waiters for the context.
  waiter_base* first_waiter_;
  
  // The end of the list of waiters for the context.
  waiter_base* last_waiter_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_COUNTING_COMPLETION_CONTEXT_HPP
