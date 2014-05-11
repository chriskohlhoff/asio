//
// impl/thread_pool.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_THREAD_POOL_HPP
#define ASIO_IMPL_THREAD_POOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/executor_op.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/task_io_service_allocator.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

inline thread_pool::executor_type thread_pool::get_executor() const
{
  return executor_type(const_cast<thread_pool&>(*this));
}

inline execution_context& thread_pool::executor_type::context()
{
  return pool_;
}

template <typename Function>
void thread_pool::executor_type::dispatch(ASIO_MOVE_ARG(Function) f)
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(f));

  // Invoke immediately if we are already inside the thread pool.
  if (pool_.scheduler_.can_dispatch())
  {
    detail::fenced_block b(detail::fenced_block::full);
    tmp();
    return;
  }

  // Allocate and construct an operation to wrap the function.
  typedef detail::task_io_service_allocator<void> allocator_type;
  typedef detail::executor_op<function_type, allocator_type> op;
  typename op::ptr p = { allocator_type(), 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator_type());

  ASIO_HANDLER_CREATION((p.p, "thread_pool", this, "post"));

  pool_.scheduler_.post_immediate_completion(p.p, false);
  p.v = p.p = 0;
}

template <typename Function>
void thread_pool::executor_type::post(ASIO_MOVE_ARG(Function) f)
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(f));

  // Allocate and construct an operation to wrap the function.
  typedef detail::task_io_service_allocator<void> allocator_type;
  typedef detail::executor_op<function_type, allocator_type> op;
  typename op::ptr p = { allocator_type(), 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator_type());

  ASIO_HANDLER_CREATION((p.p, "thread_pool", this, "post"));

  pool_.scheduler_.post_immediate_completion(p.p, false);
  p.v = p.p = 0;
}

template <typename Function>
void thread_pool::executor_type::defer(ASIO_MOVE_ARG(Function) f)
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(f));

  // Allocate and construct an operation to wrap the function.
  typedef detail::task_io_service_allocator<void> allocator_type;
  typedef detail::executor_op<function_type, allocator_type> op;
  typename op::ptr p = { allocator_type(), 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator_type());

  ASIO_HANDLER_CREATION((p.p, "thread_pool", this, "defer"));

  pool_.scheduler_.post_immediate_completion(p.p, true);
  p.v = p.p = 0;
}

inline thread_pool::executor_type::work::work(thread_pool::executor_type& e)
  : scheduler_(e.pool_.scheduler_)
{
  scheduler_.work_started();
}

inline thread_pool::executor_type::work::work(const work& other)
  : scheduler_(other.scheduler_)
{
  scheduler_.work_started();
}

#if defined(ASIO_HAS_MOVE)
inline thread_pool::executor_type::work::work(work&& other)
  : scheduler_(other.scheduler_)
{
  scheduler_.work_started();
}
#endif // defined(ASIO_HAS_MOVE)

inline thread_pool::executor_type::work::~work()
{
  scheduler_.work_finished();
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_THREAD_POOL_HPP
