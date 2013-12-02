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
    ops.push(impl->putters_);
    ops.push(impl->getters_);
    impl = impl->next_;
  }
  io_service_.abandon_operations(ops);
}

void channel_service::construct(
    channel_service::base_implementation_type& impl,
    std::size_t max_buffer_size)
{
  impl.max_buffer_size_ = max_buffer_size;

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
  close(impl);
}

void channel_service::close(channel_service::base_implementation_type& impl)
{
  impl.open_ = false;
  detail::op_queue<detail::operation> ops; // TODO broken_pipe
  ops.push(impl.getters_);
  io_service_.post_deferred_completions(ops);
}

void channel_service::cancel(channel_service::base_implementation_type& impl)
{
  detail::op_queue<detail::operation> ops; // TODO operation_aborted
  ops.push(impl.putters_);
  ops.push(impl.getters_);
  io_service_.post_deferred_completions(ops);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_CHANNEL_SERVICE_IPP
