//
// impl/dispatch.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_DISPATCH_HPP
#define ASIO_IMPL_DISPATCH_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associated_allocator.hpp"
#include "asio/associated_executor.hpp"
#include "asio/detail/work_dispatcher.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Executor, typename Allocator, typename Handler>
inline void dispatch(const Executor& ex,
    const Allocator& alloc, Handler& handler,
    typename enable_if<!execution::can_require<Executor,
      execution::oneway_t, execution::single_t>::value>::type* = 0)
{
  ex.dispatch(ASIO_MOVE_CAST(Handler)(handler), alloc);
}

template <typename Executor, typename Allocator, typename Handler>
inline void dispatch(const Executor& ex,
    const Allocator& alloc, Handler& handler,
    typename enable_if<execution::can_require<Executor,
      execution::oneway_t, execution::single_t>::value>::type* = 0)
{
  execution::prefer(
      execution::require(ex, execution::oneway, execution::single),
      execution::blocking.possibly, execution::allocator(alloc)
    ).execute(ASIO_MOVE_CAST(Handler)(handler));
}

template <typename Executor, typename Allocator, typename Handler>
inline void dispatch_work(const Executor& ex,
    const Allocator& alloc, Handler& handler,
    typename enable_if<!execution::can_require<Executor,
      execution::oneway_t, execution::single_t>::value>::type* = 0)
{
  ex.dispatch(detail::work_dispatcher<Handler>(handler), alloc);
}

template <typename Executor, typename Allocator, typename Handler>
inline void dispatch_work(const Executor& ex,
    const Allocator& alloc, Handler& handler,
    typename enable_if<execution::can_require<Executor,
      execution::oneway_t, execution::single_t>::value>::type* = 0)
{
  execution::prefer(
      execution::require(ex, execution::oneway, execution::single),
      execution::blocking.possibly, execution::allocator(alloc)
    ).execute(detail::work_dispatcher<Handler>(handler));
}

} // namespace detail

template <typename CompletionToken>
ASIO_INITFN_RESULT_TYPE(CompletionToken, void()) dispatch(
    ASIO_MOVE_ARG(CompletionToken) token)
{
  typedef ASIO_HANDLER_TYPE(CompletionToken, void()) handler;

  async_completion<CompletionToken, void()> init(token);

  typename associated_executor<handler>::type ex(
      (get_associated_executor)(init.completion_handler));

  typename associated_allocator<handler>::type alloc(
      (get_associated_allocator)(init.completion_handler));

  detail::dispatch(ex, alloc, init.completion_handler);

  return init.result.get();
}

template <typename Executor, typename CompletionToken>
ASIO_INITFN_RESULT_TYPE(CompletionToken, void()) dispatch(
    const Executor& ex, ASIO_MOVE_ARG(CompletionToken) token,
    typename enable_if<!is_convertible<
      Executor&, execution_context&>::value>::type*)
{
  typedef ASIO_HANDLER_TYPE(CompletionToken, void()) handler;

  async_completion<CompletionToken, void()> init(token);

  typename associated_allocator<handler>::type alloc(
      (get_associated_allocator)(init.completion_handler));

  detail::dispatch_work(ex, alloc, init.completion_handler);

  return init.result.get();
}

template <typename ExecutionContext, typename CompletionToken>
inline ASIO_INITFN_RESULT_TYPE(CompletionToken, void()) dispatch(
    ExecutionContext& ctx, ASIO_MOVE_ARG(CompletionToken) token,
    typename enable_if<is_convertible<
      ExecutionContext&, execution_context&>::value>::type*)
{
  return (dispatch)(ctx.get_executor(),
      ASIO_MOVE_CAST(CompletionToken)(token));
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_DISPATCH_HPP
