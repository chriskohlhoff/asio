//
// impl/system_executor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_SYSTEM_EXECUTOR_HPP
#define ASIO_IMPL_SYSTEM_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include <vector>
#include "asio/detail/assert.hpp"
#include "asio/detail/executor_op.hpp"
#include "asio/detail/global.hpp"
#include "asio/detail/shared_ptr.hpp"
#include "asio/detail/static_mutex.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/task_io_service.hpp"
#include "asio/detail/task_io_service_allocator.hpp"
#include "asio/detail/thread.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct system_context_base
  : public execution_context
{
public:
  task_io_service& scheduler_;

  system_context_base()
    : scheduler_(use_service<task_io_service>(*this))
  {
    scheduler_.work_started();
  }

  ~system_context_base()
  {
    scheduler_.work_finished();

    scheduler_.stop();
    for (std::size_t i = 0; i < threads_.size(); ++i)
      threads_[i]->join();

    shutdown_context();
    destroy_context();
  }

protected:
  void start_threads()
  {
    std::size_t n = thread::hardware_concurrency() * 2;
    if (n == 0)
      n = 2;

    for (std::size_t i = 0; i < n; ++i)
    {
      thread_function f = { &scheduler_ };
      shared_ptr<thread> t(new thread(f));
      threads_.push_back(t);
    }
  }

private:
  struct thread_function
  {
    task_io_service* scheduler_;

    void operator()()
    {
      asio::error_code ec;
      scheduler_->run(ec);
    }
  };

  std::vector<shared_ptr<thread> > threads_;
};

class system_context
  : public system_context_base
{
public:
  system_context()
  {
    start_threads();
  }
};

} // namespace detail

execution_context& system_executor::context()
{
  return detail::global<detail::system_context>();
}

template <typename Function>
void system_executor::dispatch(ASIO_MOVE_ARG(Function) f)
{
  typename decay<Function>::type tmp(ASIO_MOVE_CAST(Function)(f));
  tmp();
}

template <typename Function>
void system_executor::post(ASIO_MOVE_ARG(Function) f)
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

  ASIO_HANDLER_CREATION((p.p, "system_executor", this, "post"));

  detail::system_context& ctx = detail::global<detail::system_context>();
  ctx.scheduler_.post_immediate_completion(p.p, false);
  p.v = p.p = 0;
}

template <typename Function>
void system_executor::defer(ASIO_MOVE_ARG(Function) f)
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

  ASIO_HANDLER_CREATION((p.p, "system_executor", this, "defer"));

  detail::system_context& ctx = detail::global<detail::system_context>();
  ctx.scheduler_.post_immediate_completion(p.p, true);
  p.v = p.p = 0;
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_SYSTEM_EXECUTOR_HPP
