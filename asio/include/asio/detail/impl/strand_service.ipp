//
// detail/impl/strand_service.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_STRAND_SERVICE_IPP
#define ASIO_DETAIL_IMPL_STRAND_SERVICE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/call_stack.hpp"
#include "asio/detail/strand_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct strand_service::on_do_complete_exit
{
  io_service_impl* owner_;
  strand_impl* impl_;

  ~on_do_complete_exit()
  {
    impl_->mutex_.lock();
    bool more_handlers = (--impl_->count_ > 0);
    impl_->mutex_.unlock();

    if (more_handlers)
      owner_->post_immediate_completion(impl_);
  }
};

strand_service::strand_service(asio::io_service& io_service)
  : asio::detail::service_base<strand_service>(io_service),
    io_service_(asio::use_service<io_service_impl>(io_service)),
    mutex_(),
    salt_(0)
{
}

void strand_service::shutdown_service()
{
  op_queue<operation> ops;

  asio::detail::mutex::scoped_lock lock(mutex_);

  for (std::size_t i = 0; i < num_implementations; ++i)
    if (strand_impl* impl = implementations_[i].get())
      ops.push(impl->queue_);
}

void strand_service::construct(strand_service::implementation_type& impl)
{
  std::size_t salt = salt_++;
  std::size_t index = reinterpret_cast<std::size_t>(&impl);
  index += (reinterpret_cast<std::size_t>(&impl) >> 3);
  index ^= salt + 0x9e3779b9 + (index << 6) + (index >> 2);
  index = index % num_implementations;

  asio::detail::mutex::scoped_lock lock(mutex_);

  if (!implementations_[index])
    implementations_[index].reset(new strand_impl);
  impl = implementations_[index].get();
}

void strand_service::do_complete(io_service_impl* owner, operation* base,
    asio::error_code /*ec*/, std::size_t /*bytes_transferred*/)
{
  if (owner)
  {
    strand_impl* impl = static_cast<strand_impl*>(base);

    // Get the next handler to be executed.
    impl->mutex_.lock();
    operation* o = impl->queue_.front();
    impl->queue_.pop();
    impl->mutex_.unlock();

    // Indicate that this strand is executing on the current thread.
    call_stack<strand_impl>::context ctx(impl);

    // Ensure the next handler, if any, is scheduled on block exit.
    on_do_complete_exit on_exit = { owner, impl };
    (void)on_exit;

    o->complete(*owner);
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_STRAND_SERVICE_IPP
