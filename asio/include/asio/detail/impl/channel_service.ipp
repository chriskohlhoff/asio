//
// detail/impl/channel_service.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_CHANNEL_SERVICE_IPP
#define ASIO_DETAIL_IMPL_CHANNEL_SERVICE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/channel_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

channel_service::channel_service(asio::io_service& io_service)
  : asio::detail::service_base<channel_service>(io_service),
    io_service_(use_service<io_service_impl>(io_service)),
    mutex_(),
    impl_list_(0)
{
}

void channel_service::shutdown_service()
{
  // Abandon all pending operations.
  op_queue<operation> ops;
  asio::detail::mutex::scoped_lock lock(mutex_);
  base_implementation_type* impl = impl_list_;
  while (impl)
  {
    ops.push(impl->waiters_);
    impl = impl->next_;
  }
  io_service_.abandon_operations(ops);
}

void channel_service::construct(
    channel_service::base_implementation_type& impl,
    std::size_t max_buffer_size)
{
  impl.max_buffer_size_ = max_buffer_size;
  impl.state_ = max_buffer_size ? get_block_put_buffer : get_block_put_block;

  // Insert implementation into linked list of all implementations.
  asio::detail::mutex::scoped_lock lock(mutex_);
  impl.next_ = impl_list_;
  impl.prev_ = 0;
  if (impl_list_)
    impl_list_->prev_ = &impl;
  impl_list_ = &impl;
}

void channel_service::destroy(
    channel_service::base_implementation_type& impl)
{
  cancel(impl);

  // Remove implementation from linked list of all implementations.
  asio::detail::mutex::scoped_lock lock(mutex_);
  if (impl_list_ == &impl)
    impl_list_ = impl.next_;
  if (impl.prev_)
    impl.prev_->next_ = impl.next_;
  if (impl.next_)
    impl.next_->prev_= impl.prev_;
  impl.next_ = 0;
  impl.prev_ = 0;
}

void channel_service::close(channel_service::base_implementation_type& impl)
{
#if 0
    impl.state_ = base_implementation_type::get_waiter_
    op_queue<operation> ops;
    while (channel_op_base* op = impl.waiters_.front())
    {
      impl.waiters_.pop();
      op->on_close();
      ops.push(op);
    }
    io_service_.post_deferred_completions(ops);
  }
#endif
}

void channel_service::cancel(channel_service::base_implementation_type& impl)
{
  op_queue<operation> ops;
  while (channel_op_base* op = impl.waiters_.front())
  {
    impl.waiters_.pop();
    op->on_cancel();
    ops.push(op);
  }
  io_service_.post_deferred_completions(ops);
}

void channel_service::put(implementation_type<void>& impl,
    asio::error_code& ec)
{
#if 0
  if (!impl.open_)
  {
    ec = asio::error::broken_pipe;
  }
  else if (channel_op<void>* getter =
      static_cast<channel_op<void>*>(impl.getters_.front()))
  {
    getter->set_value();
    impl.getters_.pop();
    io_service_.post_deferred_completion(getter);
    ec = asio::error_code();
  }
  else if (impl.buffered_ < impl.max_buffer_size_)
  {
    ++impl.buffered_;
    ec = asio::error_code();
  }
  else
  {
    ec = asio::error::would_block;
  }
#endif
  switch (impl.state_)
  {
  case get_block_put_block:
  case get_buffer_put_block:
  case get_waiter_put_block:
    {
      ec = asio::error::would_block;
      break;
    }
  case get_block_put_buffer:
  case get_buffer_put_buffer:
    {
      if (++impl.buffered_ == impl.max_buffer_size_)
        impl.state_ = get_buffer_put_block;
      else
        impl.state_ = get_buffer_put_buffer;
      ec = asio::error_code();
      break;
    }
  case get_block_put_waiter:
    {
      channel_op_base* getter = impl.waiters_.front();
      static_cast<channel_op<void>*>(getter)->set_value();
      impl.waiters_.pop();
      io_service_.post_deferred_completion(getter);
      ec = asio::error_code();
      break;
    }
  case get_buffer_put_closed:
  case get_waiter_put_closed:
  case closed:
  default:
    {
      ec = asio::error::broken_pipe;
      break;
    }
  }
}

void channel_service::get(implementation_type<void>& impl,
    asio::error_code& ec)
{
#if 0
  if (impl.buffered_ > 0)
  {
    if (channel_op<void>* putter =
        static_cast<channel_op<void>*>(impl.putters_.front()))
    {
      impl.putters_.pop();
      io_service_.post_deferred_completion(putter);
    }
    else
    {
      --impl.buffered_;
    }
    ec = asio::error_code();
  }
  else if (channel_op<void>* putter =
      static_cast<channel_op<void>*>(impl.putters_.front()))
  {
    impl.putters_.pop();
    io_service_.post_deferred_completion(putter);
    ec = asio::error_code();
  }
  else if (impl.open_)
  {
    ec = asio::error::would_block;
  }
  else
  {
    ec = asio::error::broken_pipe;
  }
#endif
  switch (impl.state_)
  {
  case get_block_put_block:
  case get_block_put_buffer:
  case get_block_put_waiter:
    {
      ec = asio::error::would_block;
      break;
    }
  case get_buffer_put_buffer:
    {
      if (--impl.buffered_ == 0)
        impl.state_ = get_block_put_buffer;
      ec = asio::error_code();
    }
  case get_buffer_put_block:
    {
      if (channel_op_base* putter = impl.waiters_.front())
      {
        impl.waiters_.pop();
        io_service_.post_deferred_completion(putter);
      }
      else if (--impl.buffered_ == 0)
        impl.state_ = get_block_put_buffer;
      else
        impl.state_ = get_buffer_put_buffer;
      break;
    }
  case get_buffer_put_closed:
    {
      if (channel_op_base* putter = impl.waiters_.front())
      {
        impl.waiters_.pop();
        io_service_.post_deferred_completion(putter);
      }
      else if (--impl.buffered_ == 0)
        impl.state_ = closed;
      break;
    }
  case get_waiter_put_block:
    {
      channel_op_base* putter = impl.waiters_.front();
      impl.waiters_.pop();
      if (impl.waiters_.front() == 0)
        impl.state_ = get_block_put_block;
      io_service_.post_deferred_completion(putter);
      break;
    }
  case get_waiter_put_closed:
    {
      channel_op_base* putter = impl.waiters_.front();
      impl.waiters_.pop();
      if (impl.waiters_.front() == 0)
        impl.state_ = closed;
      io_service_.post_deferred_completion(putter);
      break;
    }
  case closed:
  default:
    {
      ec = asio::error::broken_pipe;
      break;
    }
  }
}

void channel_service::start_put_op(implementation_type<void>& impl,
    channel_op<void>* putter, bool is_continuation)
{
#if 0
  if (!impl.open_)
  {
    putter->on_close();
    io_service_.post_immediate_completion(putter, is_continuation);
  }
  else if (channel_op<void>* getter =
      static_cast<channel_op<void>*>(impl.getters_.front()))
  {
    getter->set_value();
    impl.getters_.pop();
    io_service_.post_deferred_completion(getter);
    io_service_.post_immediate_completion(putter, is_continuation);
  }
  else
  {
    if (impl.buffered_ < impl.max_buffer_size_)
    {
      ++impl.buffered_;
      io_service_.post_immediate_completion(putter, is_continuation);
    }
    else
    {
      impl.putters_.push(putter);
      io_service_.work_started();
    }
  }
#endif
#if 0
  switch (impl.state_)
  {
  case get_buffer_put_block:
    {
      impl.waiters_.push(putter);
      io_service_.work_started();
      impl.state_ = get_buffer_put_block;
      break;
    }
  case get_block_put_block:
  case get_waiter_put_block:
    {
      impl.waiters_.push(putter);
      io_service_.work_started();
      impl.state_ = get_waiter_put_block;
      break;
    }
  case get_block_put_buffer:
  case get_buffer_put_buffer:
    {
      if (++impl.buffered_ == impl.max_buffer_size_)
        impl.state_ = get_buffer_put_block;
      else
        impl.state_ = get_buffer_put_buffer;
      ec = asio::error_code();
      break;
    }
  case get_block_put_waiter:
    {
      channel_op_base* getter = impl.waiters_.front();
      static_cast<channel_op<void>*>(getter)->set_value();
      impl.waiters_.pop();
      io_service_.post_deferred_completion(getter);
      ec = asio::error_code();
      break;
    }
  case get_buffer_put_closed:
  case get_waiter_put_closed:
  case closed:
  default:
    {
      ec = asio::error::broken_pipe;
      break;
    }
  }
#endif
}

void channel_service::start_get_op(implementation_type<void>& impl,
    channel_op<void>* getter, bool is_continuation)
{
#if 0
  if (impl.buffered_ > 0)
  {
    getter->set_value();
    if (channel_op<void>* putter =
        static_cast<channel_op<void>*>(impl.putters_.front()))
    {
      impl.putters_.pop();
      io_service_.post_deferred_completion(putter);
    }
    else
    {
      --impl.buffered_;
    }
    io_service_.post_immediate_completion(getter, is_continuation);
  }
  else if (channel_op<void>* putter =
      static_cast<channel_op<void>*>(impl.putters_.front()))
  {
    getter->set_value();
    impl.putters_.pop();
    io_service_.post_deferred_completion(putter);
    io_service_.post_immediate_completion(getter, is_continuation);
  }
  else if (impl.open_)
  {
    impl.getters_.push(getter);
    io_service_.work_started();
  }
  else
  {
    getter->on_close();
    io_service_.post_immediate_completion(getter, is_continuation);
  }
#endif
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_CHANNEL_SERVICE_IPP
