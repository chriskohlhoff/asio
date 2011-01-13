//
// detail/impl/reactive_descriptor_service.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_REACTIVE_DESCRIPTOR_SERVICE_IPP
#define ASIO_DETAIL_IMPL_REACTIVE_DESCRIPTOR_SERVICE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#include "asio/error.hpp"
#include "asio/detail/reactive_descriptor_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

reactive_descriptor_service::reactive_descriptor_service(
    asio::io_service& io_service)
  : reactor_(asio::use_service<reactor>(io_service))
{
  reactor_.init_task();
}

void reactive_descriptor_service::shutdown_service()
{
}

void reactive_descriptor_service::construct(
    reactive_descriptor_service::implementation_type& impl)
{
  impl.descriptor_ = -1;
  impl.state_ = 0;
}

void reactive_descriptor_service::destroy(
    reactive_descriptor_service::implementation_type& impl)
{
  if (is_open(impl))
    reactor_.close_descriptor(impl.descriptor_, impl.reactor_data_);

  asio::error_code ignored_ec;
  descriptor_ops::close(impl.descriptor_, impl.state_, ignored_ec);
}

asio::error_code reactive_descriptor_service::assign(
    reactive_descriptor_service::implementation_type& impl,
    const native_type& native_descriptor, asio::error_code& ec)
{
  if (is_open(impl))
  {
    ec = asio::error::already_open;
    return ec;
  }

  if (int err = reactor_.register_descriptor(
        native_descriptor, impl.reactor_data_))
  {
    ec = asio::error_code(err,
        asio::error::get_system_category());
    return ec;
  }

  impl.descriptor_ = native_descriptor;
  impl.state_ = 0;
  ec = asio::error_code();
  return ec;
}

asio::error_code reactive_descriptor_service::close(
    reactive_descriptor_service::implementation_type& impl,
    asio::error_code& ec)
{
  if (is_open(impl))
    reactor_.close_descriptor(impl.descriptor_, impl.reactor_data_);

  if (descriptor_ops::close(impl.descriptor_, impl.state_, ec) == 0)
    construct(impl);

  return ec;
}

asio::error_code reactive_descriptor_service::cancel(
    reactive_descriptor_service::implementation_type& impl,
    asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  reactor_.cancel_ops(impl.descriptor_, impl.reactor_data_);
  ec = asio::error_code();
  return ec;
}

void reactive_descriptor_service::start_op(
    reactive_descriptor_service::implementation_type& impl,
    int op_type, reactor_op* op, bool non_blocking, bool noop)
{
  if (!noop)
  {
    if ((impl.state_ & descriptor_ops::non_blocking) ||
        descriptor_ops::set_internal_non_blocking(
          impl.descriptor_, impl.state_, op->ec_))
    {
      reactor_.start_op(op_type, impl.descriptor_,
          impl.reactor_data_, op, non_blocking);
      return;
    }
  }

  reactor_.post_immediate_completion(op);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#endif // ASIO_DETAIL_IMPL_REACTIVE_DESCRIPTOR_SERVICE_IPP
