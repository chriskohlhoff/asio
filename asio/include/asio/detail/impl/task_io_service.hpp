//
// detail/impl/task_io_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_TASK_IO_SERVICE_HPP
#define ASIO_DETAIL_IMPL_TASK_IO_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/task_io_service.hpp"
#include "asio/detail/call_stack.hpp"
#include "asio/detail/completion_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"

namespace asio {
namespace detail {

inline void task_io_service::work_started()
{
  ++outstanding_work_;
}

inline void task_io_service::work_finished()
{
  if (--outstanding_work_ == 0)
    stop();
}

template <typename Handler>
void task_io_service::dispatch(Handler handler)
{
  if (call_stack<task_io_service>::contains(this))
  {
    asio::detail::fenced_block b;
    asio_handler_invoke_helpers::invoke(handler, handler);
  }
  else
    post(handler);
}

template <typename Handler>
void task_io_service::post(Handler handler)
{
  // Allocate and construct an operation to wrap the handler.
  typedef completion_handler<Handler> value_type;
  typedef handler_alloc_traits<Handler, value_type> alloc_traits;
  raw_handler_ptr<alloc_traits> raw_ptr(handler);
  handler_ptr<alloc_traits> ptr(raw_ptr, handler);

  post_immediate_completion(ptr.get());
  ptr.release();
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_TASK_IO_SERVICE_HPP
