//
// impl/thread_pool.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_THREAD_POOL_HPP
#define ASIO_IMPL_THREAD_POOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "../detail/executor_op.hpp"
#include "../detail/fenced_block.hpp"
#include "../detail/recycling_allocator.hpp"
#include "../detail/type_traits.hpp"
#include "../execution_context.hpp"

#include "../detail/push_options.hpp"

namespace asio {

inline thread_pool::executor_type
thread_pool::get_executor() ASIO_NOEXCEPT
{
  return executor_type(*this);
}

inline thread_pool&
thread_pool::executor_type::context() const ASIO_NOEXCEPT
{
  return pool_;
}

inline void
thread_pool::executor_type::on_work_started() const ASIO_NOEXCEPT
{
  pool_.scheduler_.work_started();
}

inline void thread_pool::executor_type::on_work_finished()
const ASIO_NOEXCEPT
{
  pool_.scheduler_.work_finished();
}

template <typename Function, typename Allocator>
void thread_pool::executor_type::dispatch(
    ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(f));

  // Invoke immediately if we are already inside the thread pool.
  if (pool_.scheduler_.can_dispatch())
  {
    detail::fenced_block b(detail::fenced_block::full);
    asio_handler_invoke_helpers::invoke(tmp, tmp);
    return;
  }

  // Construct an allocator to be used for the operation.
  typedef typename detail::get_recycling_allocator<Allocator>::type alloc_type;
  alloc_type allocator(detail::get_recycling_allocator<Allocator>::get(a));

  // Allocate and construct an operation to wrap the function.
  typedef detail::executor_op<function_type, alloc_type> op;
  typename op::ptr p = { allocator, 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator);

  ASIO_HANDLER_CREATION((pool_, *p.p,
        "thread_pool", &this->context(), 0, "dispatch"));

  pool_.scheduler_.post_immediate_completion(p.p, false);
  p.v = p.p = 0;
}

template <typename Function, typename Allocator>
void thread_pool::executor_type::post(
    ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(f));

  // Construct an allocator to be used for the operation.
  typedef typename detail::get_recycling_allocator<Allocator>::type alloc_type;
  alloc_type allocator(detail::get_recycling_allocator<Allocator>::get(a));

  // Allocate and construct an operation to wrap the function.
  typedef detail::executor_op<function_type, alloc_type> op;
  typename op::ptr p = { allocator, 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator);

  ASIO_HANDLER_CREATION((pool_, *p.p,
        "thread_pool", &this->context(), 0, "post"));

  pool_.scheduler_.post_immediate_completion(p.p, false);
  p.v = p.p = 0;
}

template <typename Function, typename Allocator>
void thread_pool::executor_type::defer(
    ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(f));

  // Construct an allocator to be used for the operation.
  typedef typename detail::get_recycling_allocator<Allocator>::type alloc_type;
  alloc_type allocator(detail::get_recycling_allocator<Allocator>::get(a));

  // Allocate and construct an operation to wrap the function.
  typedef detail::executor_op<function_type, alloc_type> op;
  typename op::ptr p = { allocator, 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator);

  ASIO_HANDLER_CREATION((pool_, *p.p,
        "thread_pool", &this->context(), 0, "defer"));

  pool_.scheduler_.post_immediate_completion(p.p, true);
  p.v = p.p = 0;
}

inline bool
thread_pool::executor_type::running_in_this_thread() const ASIO_NOEXCEPT
{
  return pool_.scheduler_.can_dispatch();
}

} // namespace asio

#include "../detail/pop_options.hpp"

#endif // ASIO_IMPL_THREAD_POOL_HPP
