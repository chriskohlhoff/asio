//
// impl/io_context.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_IO_CONTEXT_HPP
#define ASIO_IMPL_IO_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "../detail/completion_handler.hpp"
#include "../detail/executor_op.hpp"
#include "../detail/fenced_block.hpp"
#include "../detail/handler_type_requirements.hpp"
#include "../detail/recycling_allocator.hpp"
#include "../detail/service_registry.hpp"
#include "../detail/type_traits.hpp"

#include "../detail/push_options.hpp"

namespace asio {

template <typename Service>
inline Service& use_service(io_context& ioc)
{
  // Check that Service meets the necessary type requirements.
  (void)static_cast<execution_context::service*>(static_cast<Service*>(0));
  (void)static_cast<const execution_context::id*>(&Service::id);

  return ioc.service_registry_->template use_service<Service>(ioc);
}

template <>
inline detail::io_context_impl& use_service<detail::io_context_impl>(
    io_context& ioc)
{
  return ioc.impl_;
}

} // namespace asio

#include "../detail/pop_options.hpp"

#if defined(ASIO_HAS_IOCP)
# include "../detail/win_iocp_io_context.hpp"
#else
# include "../detail/scheduler.hpp"
#endif

#include "../detail/push_options.hpp"

namespace asio {

inline io_context::executor_type
io_context::get_executor() ASIO_NOEXCEPT
{
  return executor_type(*this);
}

#if !defined(ASIO_NO_DEPRECATED)

inline void io_context::reset()
{
  restart();
}

template <typename CompletionHandler>
ASIO_INITFN_RESULT_TYPE(CompletionHandler, void ())
io_context::dispatch(ASIO_MOVE_ARG(CompletionHandler) handler)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a CompletionHandler.
  ASIO_COMPLETION_HANDLER_CHECK(CompletionHandler, handler) type_check;

  async_completion<CompletionHandler, void ()> init(handler);

  if (impl_.can_dispatch())
  {
    detail::fenced_block b(detail::fenced_block::full);
    asio_handler_invoke_helpers::invoke(init.handler, init.handler);
  }
  else
  {
    // Allocate and construct an operation to wrap the handler.
    typedef detail::completion_handler<
      typename handler_type<CompletionHandler, void ()>::type> op;
    typename op::ptr p = { detail::addressof(init.handler),
      op::ptr::allocate(init.handler), 0 };
    p.p = new (p.v) op(init.handler);

    ASIO_HANDLER_CREATION((*this, *p.p,
          "io_context", this, 0, "dispatch"));

    impl_.do_dispatch(p.p);
    p.v = p.p = 0;
  }

  return init.result.get();
}

template <typename CompletionHandler>
ASIO_INITFN_RESULT_TYPE(CompletionHandler, void ())
io_context::post(ASIO_MOVE_ARG(CompletionHandler) handler)
{
  // If you get an error on the following line it means that your handler does
  // not meet the documented type requirements for a CompletionHandler.
  ASIO_COMPLETION_HANDLER_CHECK(CompletionHandler, handler) type_check;

  async_completion<CompletionHandler, void ()> init(handler);

  bool is_continuation =
    asio_handler_cont_helpers::is_continuation(init.handler);

  // Allocate and construct an operation to wrap the handler.
  typedef detail::completion_handler<
    typename handler_type<CompletionHandler, void ()>::type> op;
  typename op::ptr p = { detail::addressof(init.handler),
      op::ptr::allocate(init.handler), 0 };
  p.p = new (p.v) op(init.handler);

  ASIO_HANDLER_CREATION((*this, *p.p,
        "io_context", this, 0, "post"));

  impl_.post_immediate_completion(p.p, is_continuation);
  p.v = p.p = 0;

  return init.result.get();
}

template <typename Handler>
#if defined(GENERATING_DOCUMENTATION)
unspecified
#else
inline detail::wrapped_handler<io_context&, Handler>
#endif
io_context::wrap(Handler handler)
{
  return detail::wrapped_handler<io_context&, Handler>(*this, handler);
}

#endif // !defined(ASIO_NO_DEPRECATED)

inline io_context&
io_context::executor_type::context() const ASIO_NOEXCEPT
{
  return io_context_;
}

inline void
io_context::executor_type::on_work_started() const ASIO_NOEXCEPT
{
  io_context_.impl_.work_started();
}

inline void
io_context::executor_type::on_work_finished() const ASIO_NOEXCEPT
{
  io_context_.impl_.work_finished();
}

template <typename Function, typename Allocator>
void io_context::executor_type::dispatch(
    ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(f));

  // Invoke immediately if we are already inside the thread pool.
  if (io_context_.impl_.can_dispatch())
  {
    detail::fenced_block b(detail::fenced_block::full);
    asio_handler_invoke_helpers::invoke(tmp, tmp);
    return;
  }

  // Construct an allocator to be used for the operation.
  typedef typename detail::get_recycling_allocator<Allocator>::type alloc_type;
  alloc_type allocator(detail::get_recycling_allocator<Allocator>::get(a));

  // Allocate and construct an operation to wrap the function.
  typedef detail::executor_op<function_type, alloc_type, detail::operation> op;
  typename op::ptr p = { allocator, 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator);

  ASIO_HANDLER_CREATION((this->context(), *p.p,
        "io_context", &this->context(), 0, "post"));

  io_context_.impl_.post_immediate_completion(p.p, false);
  p.v = p.p = 0;
}

template <typename Function, typename Allocator>
void io_context::executor_type::post(
    ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(f));

  // Construct an allocator to be used for the operation.
  typedef typename detail::get_recycling_allocator<Allocator>::type alloc_type;
  alloc_type allocator(detail::get_recycling_allocator<Allocator>::get(a));

  // Allocate and construct an operation to wrap the function.
  typedef detail::executor_op<function_type, alloc_type, detail::operation> op;
  typename op::ptr p = { allocator, 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator);

  ASIO_HANDLER_CREATION((this->context(), *p.p,
        "io_context", &this->context(), 0, "post"));

  io_context_.impl_.post_immediate_completion(p.p, false);
  p.v = p.p = 0;
}

template <typename Function, typename Allocator>
void io_context::executor_type::defer(
    ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
  // Make a local, non-const copy of the function.
  typedef typename decay<Function>::type function_type;
  function_type tmp(ASIO_MOVE_CAST(Function)(f));

  // Construct an allocator to be used for the operation.
  typedef typename detail::get_recycling_allocator<Allocator>::type alloc_type;
  alloc_type allocator(detail::get_recycling_allocator<Allocator>::get(a));

  // Allocate and construct an operation to wrap the function.
  typedef detail::executor_op<function_type, alloc_type, detail::operation> op;
  typename op::ptr p = { allocator, 0, 0 };
  p.v = p.a.allocate(1);
  p.p = new (p.v) op(tmp, allocator);

  ASIO_HANDLER_CREATION((this->context(), *p.p,
        "io_context", &this->context(), 0, "defer"));

  io_context_.impl_.post_immediate_completion(p.p, true);
  p.v = p.p = 0;
}

inline bool
io_context::executor_type::running_in_this_thread() const ASIO_NOEXCEPT
{
  return io_context_.impl_.can_dispatch();
}

inline io_context::work::work(asio::io_context& io_context)
  : io_context_impl_(io_context.impl_)
{
  io_context_impl_.work_started();
}

inline io_context::work::work(const work& other)
  : io_context_impl_(other.io_context_impl_)
{
  io_context_impl_.work_started();
}

inline io_context::work::~work()
{
  io_context_impl_.work_finished();
}

inline asio::io_context& io_context::work::get_io_context()
{
  return static_cast<asio::io_context&>(io_context_impl_.context());
}

#if !defined(ASIO_NO_DEPRECATED)
inline asio::io_context& io_context::work::get_io_service()
{
  return static_cast<asio::io_context&>(io_context_impl_.context());
}
#endif // !defined(ASIO_NO_DEPRECATED)

inline asio::io_context& io_context::service::get_io_context()
{
  return static_cast<asio::io_context&>(context());
}

#if !defined(ASIO_NO_DEPRECATED)
inline asio::io_context& io_context::service::get_io_service()
{
  return static_cast<asio::io_context&>(context());
}
#endif // !defined(ASIO_NO_DEPRECATED)

} // namespace asio

#include "../detail/pop_options.hpp"

#endif // ASIO_IMPL_IO_CONTEXT_HPP
