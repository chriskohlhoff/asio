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
  impl.get_state_ = block;
  impl.put_state_ = max_buffer_size ? buffer : block;

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
  impl.put_state_ = closed;
  switch (impl.get_state_)
  {
  case block:
    {
      impl.get_state_ = closed;
      op_queue<operation> ops;
      while (channel_op_base* op = impl.waiters_.front())
      {
        impl.waiters_.pop();
        op->on_close();
        ops.push(op);
      }
      io_service_.post_deferred_completions(ops);
      break;
    }
  case buffer:
  case waiter:
  default:
    break;
  }
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

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_CHANNEL_SERVICE_IPP
