//
// strand_service.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_STRAND_SERVICE_HPP
#define ASIO_DETAIL_STRAND_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/functional/hash.hpp>
#include <boost/scoped_ptr.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/io_service.hpp"
#include "asio/detail/call_stack.hpp"
#include "asio/detail/completion_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/operation.hpp"
#include "asio/detail/service_base.hpp"

namespace asio {
namespace detail {

// Default service implementation for a strand.
class strand_service
  : public asio::detail::service_base<strand_service>
{
private:
  struct on_do_complete_exit;
  struct on_dispatch_exit;

public:

  // The underlying implementation of a strand.
  class strand_impl
    : public operation
  {
  public:
    strand_impl()
      : operation(&strand_service::do_complete),
        count_(0)
    {
    }

  private:
    // Only this service will have access to the internal values.
    friend class strand_service;
    friend struct on_do_complete_exit;
    friend struct on_dispatch_exit;

    // Mutex to protect access to internal data.
    asio::detail::mutex mutex_;

    // The count of handlers in the strand, including the upcall (if any).
    std::size_t count_;

    // The handlers waiting on the strand.
    op_queue<operation> queue_;
  };

  typedef strand_impl* implementation_type;

  // Construct a new strand service for the specified io_service.
  explicit strand_service(asio::io_service& io_service)
    : asio::detail::service_base<strand_service>(io_service),
      io_service_(asio::use_service<io_service_impl>(io_service)),
      mutex_(),
      salt_(0)
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    op_queue<operation> ops;

    asio::detail::mutex::scoped_lock lock(mutex_);

    for (std::size_t i = 0; i < num_implementations; ++i)
      if (strand_impl* impl = implementations_[i].get())
        ops.push(impl->queue_);
  }

  // Construct a new strand implementation.
  void construct(implementation_type& impl)
  {
    std::size_t index = boost::hash_value(&impl);
    boost::hash_combine(index, salt_++);
    index = index % num_implementations;

    asio::detail::mutex::scoped_lock lock(mutex_);

    if (!implementations_[index])
      implementations_[index].reset(new strand_impl);
    impl = implementations_[index].get();
  }

  // Destroy a strand implementation.
  void destroy(implementation_type& impl)
  {
    impl = 0;
  }

  // Request the io_service to invoke the given handler.
  template <typename Handler>
  void dispatch(implementation_type& impl, Handler handler)
  {
    // If we are already in the strand then the handler can run immediately.
    if (call_stack<strand_impl>::contains(impl))
    {
      asio::detail::fenced_block b;
      asio_handler_invoke_helpers::invoke(handler, handler);
      return;
    }

    // Allocate and construct an object to wrap the handler.
    typedef completion_handler<Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, handler);

    // If we are running inside the io_service, and no other handler is queued
    // or running, then the handler can run immediately.
    bool can_dispatch = call_stack<io_service_impl>::contains(&io_service_);
    impl->mutex_.lock();
    bool first = (++impl->count_ == 1);
    if (can_dispatch && first)
    {
      // Immediate invocation is allowed.
      impl->mutex_.unlock();

      // Memory must be releaesed before any upcall is made.
      ptr.reset();

      // Indicate that this strand is executing on the current thread.
      call_stack<strand_impl>::context ctx(impl);

      // Ensure the next handler, if any, is scheduled on block exit.
      on_dispatch_exit on_exit = { &io_service_, impl };
      (void)on_exit;

      asio::detail::fenced_block b;
      asio_handler_invoke_helpers::invoke(handler, handler);
      return;
    }

    // Immediate invocation is not allowed, so enqueue for later.
    impl->queue_.push(ptr.get());
    impl->mutex_.unlock();
    ptr.release();

    // The first handler to be enqueued is responsible for scheduling the
    // strand.
    if (first)
      io_service_.post_immediate_completion(impl);
  }

  // Request the io_service to invoke the given handler and return immediately.
  template <typename Handler>
  void post(implementation_type& impl, Handler handler)
  {
    // Allocate and construct an object to wrap the handler.
    typedef completion_handler<Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, handler);

    // Add the handler to the queue.
    impl->mutex_.lock();
    bool first = (++impl->count_ == 1);
    impl->queue_.push(ptr.get());
    impl->mutex_.unlock();
    ptr.release();

    // The first handler to be enqueue is responsible for scheduling the strand.
    if (first)
      io_service_.post_immediate_completion(impl);
  }

private:
  static void do_complete(io_service_impl* owner, operation* base,
      asio::error_code /*ec*/, std::size_t /*bytes_transferred*/)
  {
    if (owner)
    {
      strand_impl* impl = static_cast<strand_impl*>(base);

      // Get the next handler to be executed.
      impl->mutex_.lock();
      operation* o = impl->queue_.front();
      impl->queue_.pop();
      impl->mutex_.unlock();

      // Indicate that this strand is executing on the current thread.
      call_stack<strand_impl>::context ctx(impl);

      // Ensure the next handler, if any, is scheduled on block exit.
      on_do_complete_exit on_exit = { owner, impl };
      (void)on_exit;

      o->complete(*owner);
    }
  }

  // Helper class to re-post the strand on exit.
  struct on_do_complete_exit
  {
    io_service_impl* owner_;
    strand_impl* impl_;

    ~on_do_complete_exit()
    {
      impl_->mutex_.lock();
      bool more_handlers = (--impl_->count_ > 0);
      impl_->mutex_.unlock();

      if (more_handlers)
        owner_->post_immediate_completion(impl_);
    }
  };

  // Helper class to re-post the strand on exit.
  struct on_dispatch_exit
  {
    io_service_impl* io_service_;
    strand_impl* impl_;

    ~on_dispatch_exit()
    {
      impl_->mutex_.lock();
      bool more_handlers = (--impl_->count_ > 0);
      impl_->mutex_.unlock();

      if (more_handlers)
        io_service_->post_immediate_completion(impl_);
    }
  };

  // The io_service implementation used to post completions.
  io_service_impl& io_service_;

  // Mutex to protect access to the array of implementations.
  asio::detail::mutex mutex_;

  // Number of implementations shared between all strand objects.
  enum { num_implementations = 193 };

  // The head of a linked list of all implementations.
  boost::scoped_ptr<strand_impl> implementations_[num_implementations];

  // Extra value used when hashing to prevent recycled memory locations from
  // getting the same strand implementation.
  std::size_t salt_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_STRAND_SERVICE_HPP
