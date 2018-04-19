//
// detail/handler_work.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include "asio/detail/is_executor.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Handler, typename Executor, typename>
class handler_work;

// A helper class template to track outstanding work associated with a handler.
// The primary template uses the new executors framework.
template <typename Handler,
    typename Executor = typename associated_executor<Handler>::type,
    typename = void>
class handler_work_outstanding
{
public:
  explicit handler_work_outstanding(Handler& handler) ASIO_NOEXCEPT
  {
    Executor ex(associated_executor<Handler>::get(handler));
    ex.on_work_started();
  }
};

// This specialisation eliminates all cost of tracking outstanding work.
template <typename Handler>
class handler_work_outstanding<Handler, system_executor>
{
public:
  explicit handler_work_outstanding(Handler&) ASIO_NOEXCEPT
  {
  }
};

// This specialisation uses the new unified executors framework.
template <typename Handler, typename Executor>
class handler_work_outstanding<Handler, Executor,
    typename enable_if<!is_executor<Executor>::value>::type>
{
public:
  explicit handler_work_outstanding(Handler& handler) ASIO_NOEXCEPT
    : executor_(execution::prefer(
          (get_associated_executor)(handler),
          execution::outstanding_work.tracked))
  {
  }

private:
  friend class handler_work<Handler, Executor, void>;

  decltype(execution::prefer(declval<Executor>(),
        execution::outstanding_work.tracked)) executor_;
};

// A helper class template to allow completion handlers to be dispatched
// through either the new executors framework or the old invocaton hook. The
// primary template uses the new executors framework.
template <typename Handler,
    typename Executor = typename associated_executor<Handler>::type,
    typename = void>
class handler_work
{
public:
  explicit handler_work(Handler& handler,
      handler_work_outstanding<Handler, Executor>&) ASIO_NOEXCEPT
    : executor_(associated_executor<Handler>::get(handler))
  {
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

  typename associated_executor<Handler>::type executor_;
};

// This specialisation dispatches a handler through the old invocation hook.
// The specialisation is not strictly required for correctness, as the
// system_executor will dispatch through the hook anyway. However, by doing
// this we avoid an extra copy of the handler.
template <typename Handler>
class handler_work<Handler, system_executor>
{
public:
  explicit handler_work(Handler&) ASIO_NOEXCEPT
  {
  }

  explicit handler_work(Handler&,
      handler_work_outstanding<Handler, system_executor>&) ASIO_NOEXCEPT
  {
  }

  static void start(Handler&) ASIO_NOEXCEPT
  {
  }

  ~handler_work()
  {
  }

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

// This specialisation uses the new unified executors framework.
template <typename Handler, typename Executor>
class handler_work<Handler, Executor,
    typename enable_if<!is_executor<Executor>::value>::type>
{
public:
  explicit handler_work(Handler&,
      handler_work_outstanding<Handler, Executor>& w) ASIO_NOEXCEPT
    : executor_(w.executor_)
  {
  }

  ~handler_work()
  {
  }

  template <typename Function>
  void complete(Function& function, Handler& handler)
  {
    execution::prefer(
        execution::require(executor_, execution::oneway, execution::single),
        execution::blocking.possibly,
        execution::allocator((get_associated_allocator)(handler))
      ).execute(ASIO_MOVE_CAST(Function)(function));
  }

private:
  // Disallow copying and assignment.
  handler_work(const handler_work&);
  handler_work& operator=(const handler_work&);

  decltype(execution::prefer(declval<Executor>(),
        execution::outstanding_work.tracked)) executor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HANDLER_WORK_HPP
