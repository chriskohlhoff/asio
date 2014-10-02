//
// detail/handler_work.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_WORK_HPP
#define ASIO_DETAIL_HANDLER_WORK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associated_executor.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// A special type meeting the Executor requirements that is used to indicate
// that a handler should be executed using the old invocation hook. The type is
// not used at runtime, but only at compile time to distinguish whether a
// handler provides its own associated executor type or not.
class hook_executor
{
public:
  execution_context& context() ASIO_NOEXCEPT
  {
    return system_executor().context();
  }

  void on_work_started() ASIO_NOEXCEPT {}
  void on_work_finished() ASIO_NOEXCEPT {}

  template <typename Function, typename Allocator>
  void dispatch(ASIO_MOVE_ARG(Function), const Allocator&) {}
  template <typename Function, typename Allocator>
  void post(ASIO_MOVE_ARG(Function), const Allocator&) {}
  template <typename Function, typename Allocator>
  void defer(ASIO_MOVE_ARG(Function), const Allocator&) {}
};

// A helper class template to allow completion handlers to be dispatched
// through either the new executors framework or the old invocaton hook. The
// primary template uses the new executors framework.
template <typename Handler, typename Executor
    = typename associated_executor<Handler, hook_executor>::type>
class handler_work
{
public:
  explicit handler_work(Handler& handler) ASIO_NOEXCEPT
    : executor_(associated_executor<Handler, hook_executor>::get(handler))
  {
  }

  static void start(Handler& handler) ASIO_NOEXCEPT
  {
    Executor ex(associated_executor<Handler, hook_executor>::get(handler));
    ex.on_work_started();
  }

  ~handler_work()
  {
    executor_.on_work_finished();
  }

  template <typename Function>
  void complete(Function& function, Handler& handler)
  {
    executor_.dispatch(ASIO_MOVE_CAST(Function)(function),
        associated_allocator<Handler>::get(handler));
  }

private:
  // Disallow copying and assignment.
  handler_work(const handler_work&);
  handler_work& operator=(const handler_work&);

  typename associated_executor<Handler, hook_executor>::type executor_;
};

// This specialisation dispatches a handler through the old invocation hook.
template <typename Handler>
class handler_work<Handler, hook_executor>
{
public:
  explicit handler_work(Handler&) ASIO_NOEXCEPT {}
  static void start(Handler&) ASIO_NOEXCEPT {}
  ~handler_work() {}

  template <typename Function>
  void complete(Function& function, Handler& handler)
  {
    asio_handler_invoke_helpers::invoke(function, handler);
  }

private:
  // Disallow copying and assignment.
  handler_work(const handler_work&);
  handler_work& operator=(const handler_work&);
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HANDLER_WORK_HPP
