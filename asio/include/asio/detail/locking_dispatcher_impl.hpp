//
// locking_dispatcher_impl.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_LOCKING_DISPATCHER_IMPL_HPP
#define ASIO_DETAIL_LOCKING_DISPATCHER_IMPL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/bind_handler.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"

namespace asio {
namespace detail {

template <typename Demuxer>
class locking_dispatcher_impl
  : private noncopyable
{
public:
  // Constructor.
  locking_dispatcher_impl()
    : first_waiter_(0),
      last_waiter_(0),
      mutex_()
  {
  }

  // Request a dispatcher to invoke the given handler.
  template <typename Handler>
  void dispatch(Demuxer& demuxer, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    if (first_waiter_ == 0)
    {
      // This handler now has the lock, so can be dispatched immediately.
      first_waiter_ = last_waiter_ = new waiter<Handler>(handler);
      lock.unlock();
      demuxer.dispatch(waiter_handler(demuxer, *this));
    }
    else
    {
      // Another waiter already holds the lock, so this handler must join
      // the list of waiters. The handler will be posted automatically when
      // its turn comes.
      last_waiter_->next_ = new waiter<Handler>(handler);
      last_waiter_ = last_waiter_->next_;
    }
  }

  // Request a dispatcher to invoke the given handler and return immediately.
  template <typename Handler>
  void post(Demuxer& demuxer, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    if (first_waiter_ == 0)
    {
      // This handler now has the lock, so can be posted immediately.
      first_waiter_ = last_waiter_ = new waiter<Handler>(handler);
      lock.unlock();
      demuxer.post(waiter_handler(demuxer, *this));
    }
    else
    {
      // Another waiter already holds the lock, so this handler must join
      // the list of waiters. The handler will be posted automatically when
      // its turn comes.
      last_waiter_->next_ = new waiter<Handler>(handler);
      last_waiter_ = last_waiter_->next_;
    }
  }

private:
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

    virtual void call() = 0;

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

    virtual void call()
    {
      handler_();
    }

  private:
    Handler handler_;
  };

  // Helper class to allow waiting handlers to be dispatched.
  class waiter_handler
  {
  public:
    waiter_handler(Demuxer& demuxer, locking_dispatcher_impl<Demuxer>& impl)
      : demuxer_(demuxer),
        impl_(impl)
    {
    }

    // Helper class to automatically enqueue next waiter on block exit. This
    // class cannot be function-local to operator() due to a linker bug in MSVC,
    // where an inline-member of a function-local class is exported by each .obj
    // file that includes the header.
    class post_next_waiter_on_exit
    {
    public:
      post_next_waiter_on_exit(waiter_handler& handler)
        : handler_(handler)
      {
      }

      ~post_next_waiter_on_exit()
      {
        handler_.post_next_waiter();
      }

    private:
      waiter_handler& handler_;
    };

    void operator()()
    {
      post_next_waiter_on_exit p(*this);

      // Call the handler.
      impl_.first_waiter_->call();
    }

    void post_next_waiter()
    {
      asio::detail::mutex::scoped_lock lock(impl_.mutex_);

      waiter_base* tmp = impl_.first_waiter_;
      impl_.first_waiter_ = impl_.first_waiter_->next_;
      if (impl_.first_waiter_)
      {
        lock.unlock();

        // Ensure the waiter is not deleted until after we have finished
        // accessing the dispatcher, since the waiter might indirectly own
        // the dispatcher and so destroying the waiter will cause the
        // dispatcher to be destroyed.
        delete tmp;

        // There is more work to do, so post this handler again.
        demuxer_.post(*this);
      }
      else
      {
        impl_.last_waiter_ = 0;

        lock.unlock();

        // Ensure the waiter is not deleted until after we have finished
        // accessing the dispatcher, since the waiter might indirectly own
        // the dispatcher and so destroying the waiter will cause the
        // dispatcher to be destroyed.
        delete tmp;
      }
    }

  private:
    Demuxer& demuxer_;
    locking_dispatcher_impl<Demuxer>& impl_;
  };

  friend class waiter_handler;

  // The start of the list of waiters for the dispatcher. If this pointer
  // is non-null then it indicates that a handler holds the lock.
  waiter_base* first_waiter_;
  
  // The end of the list of waiters for the dispatcher.
  waiter_base* last_waiter_;

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_LOCKING_DISPATCHER_IMPL_HPP
