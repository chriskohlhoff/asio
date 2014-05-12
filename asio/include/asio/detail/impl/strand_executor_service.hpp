//
// detail/impl/strand_executor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_STRAND_EXECUTOR_SERVICE_HPP
#define ASIO_DETAIL_IMPL_STRAND_EXECUTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/call_stack.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/scheduler_allocator.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Executor>
class strand_executor_service::invoker
{
public:
  invoker(const implementation_type& impl, Executor& ex)
    : impl_(impl),
      executor_(ex),
      work_(ex)
  {
  }

  invoker(const invoker& other)
    : impl_(other.impl_),
      executor_(other.executor_),
      work_(other.work_)
  {
  }

#if defined(ASIO_HAS_MOVE)
  invoker(invoker&& other)
    : impl_(ASIO_MOVE_CAST(implementation_type)(other.impl_)),
      executor_(ASIO_MOVE_CAST(Executor)(other.executor_)),
      work_(ASIO_MOVE_CAST(typename Executor::work)(other.work_))
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  struct on_invoker_exit
  {
    invoker* this_;

    ~on_invoker_exit()
    {
      this_->impl_->mutex_->lock();
      this_->impl_->ready_queue_.push(this_->impl_->waiting_queue_);
      bool more_handlers = this_->impl_->locked_ =
        !this_->impl_->ready_queue_.empty();
      this_->impl_->mutex_->unlock(); 

      if (more_handlers)
      {
        Executor ex(this_->executor_);
        ex.post(ASIO_MOVE_CAST(invoker)(*this_));
      }
    }
  };

  void operator()()
  {
    // Indicate that this strand is executing on the current thread.
    call_stack<strand_impl>::context ctx(impl_.get());

    // Ensure the next handler, if any, is scheduled on block exit.
    on_invoker_exit on_exit = { this };
    (void)on_exit;

    // Run all ready handlers. No lock is required since the ready queue is
    // accessed only within the strand.
    asio::error_code ec;
    while (scheduler_operation* o = impl_->ready_queue_.front())
    {
      impl_->ready_queue_.pop();
      o->complete(impl_.get(), ec, 0);
    }
  }

private:
  implementation_type impl_;
  Executor executor_;
  typename Executor::work work_;
};

template <typename Executor, typename Function>
void strand_executor_service::dispatch(const implementation_type& impl,
    Executor& ex, ASIO_MOVE_ARG(Function) function)
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(function));

  // If we are already in the strand then the function can run immediately.
  if (call_stack<strand_impl>::contains(impl.get()))
  {
    fenced_block b(fenced_block::full);
    tmp();
    return;
  }

  // Allocate and construct an operation to wrap the function.
  typedef scheduler_allocator<void> allocator_type;
  typedef executor_op<function_type, allocator_type> op;
  typename op::ptr p = { allocator_type(), 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator_type());

  ASIO_HANDLER_CREATION((p.p, "strand_executor", this, "dispatch"));

  // Add the function to the strand and schedule the strand if required.
  bool first = enqueue(impl, p.p);
  p.v = p.p = 0;
  if (first)
    ex.dispatch(invoker<Executor>(impl, ex));
}

// Request invocation of the given function and return immediately.
template <typename Executor, typename Function>
void strand_executor_service::post(const implementation_type& impl,
    Executor& ex, ASIO_MOVE_ARG(Function) function)
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(function));

  // Allocate and construct an operation to wrap the function.
  typedef scheduler_allocator<void> allocator_type;
  typedef executor_op<function_type, allocator_type> op;
  typename op::ptr p = { allocator_type(), 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator_type());

  ASIO_HANDLER_CREATION((p.p, "strand_executor", this, "post"));

  // Add the function to the strand and schedule the strand if required.
  bool first = enqueue(impl, p.p);
  p.v = p.p = 0;
  if (first)
    ex.post(invoker<Executor>(impl, ex));
}

// Request invocation of the given function and return immediately.
template <typename Executor, typename Function>
void strand_executor_service::defer(const implementation_type& impl,
    Executor& ex, ASIO_MOVE_ARG(Function) function)
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(function));

  // Allocate and construct an operation to wrap the function.
  typedef scheduler_allocator<void> allocator_type;
  typedef executor_op<function_type, allocator_type> op;
  typename op::ptr p = { allocator_type(), 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator_type());

  ASIO_HANDLER_CREATION((p.p, "strand_executor", this, "defer"));

  // Add the function to the strand and schedule the strand if required.
  bool first = enqueue(impl, p.p);
  p.v = p.p = 0;
  if (first)
    ex.defer(invoker<Executor>(impl, ex));
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_STRAND_EXECUTOR_SERVICE_HPP
