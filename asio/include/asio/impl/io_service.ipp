//
// impl/io_service.ipp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_IO_SERVICE_IPP
#define ASIO_IMPL_IO_SERVICE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/limits.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/io_service.hpp"
#include "asio/detail/service_registry.hpp"
#include "asio/detail/throw_error.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_io_service.hpp"
#else
# include "asio/detail/task_io_service.hpp"
#endif

namespace asio {

io_service::io_service()
  : service_registry_(new asio::detail::service_registry(*this)),
    impl_(service_registry_->use_service<impl_type>())
{
  impl_.init((std::numeric_limits<std::size_t>::max)());
}

io_service::io_service(std::size_t concurrency_hint)
  : service_registry_(new asio::detail::service_registry(*this)),
    impl_(service_registry_->use_service<impl_type>())
{
  impl_.init(concurrency_hint);
}

io_service::~io_service()
{
  delete service_registry_;
}

std::size_t io_service::run()
{
  asio::error_code ec;
  std::size_t s = impl_.run(ec);
  asio::detail::throw_error(ec);
  return s;
}

std::size_t io_service::run(asio::error_code& ec)
{
  return impl_.run(ec);
}

std::size_t io_service::run_one()
{
  asio::error_code ec;
  std::size_t s = impl_.run_one(ec);
  asio::detail::throw_error(ec);
  return s;
}

std::size_t io_service::run_one(asio::error_code& ec)
{
  return impl_.run_one(ec);
}

std::size_t io_service::poll()
{
  asio::error_code ec;
  std::size_t s = impl_.poll(ec);
  asio::detail::throw_error(ec);
  return s;
}

std::size_t io_service::poll(asio::error_code& ec)
{
  return impl_.poll(ec);
}

std::size_t io_service::poll_one()
{
  asio::error_code ec;
  std::size_t s = impl_.poll_one(ec);
  asio::detail::throw_error(ec);
  return s;
}

std::size_t io_service::poll_one(asio::error_code& ec)
{
  return impl_.poll_one(ec);
}

void io_service::stop()
{
  impl_.stop();
}

void io_service::reset()
{
  impl_.reset();
}

io_service::work::work(asio::io_service& io_service)
  : io_service_(io_service)
{
  io_service_.impl_.work_started();
}

io_service::work::work(const work& other)
  : io_service_(other.io_service_)
{
  io_service_.impl_.work_started();
}

io_service::work::~work()
{
  io_service_.impl_.work_finished();
}

asio::io_service& io_service::work::io_service()
{
  return io_service_;
}

asio::io_service& io_service::work::get_io_service()
{
  return io_service_;
}

io_service::service::service(asio::io_service& owner)
  : owner_(owner),
    next_(0)
{
}

io_service::service::~service()
{
}

asio::io_service& io_service::service::io_service()
{
  return owner_;
}

asio::io_service& io_service::service::get_io_service()
{
  return owner_;
}

service_already_exists::service_already_exists()
  : std::logic_error("Service already exists.")
{
}

invalid_service_owner::invalid_service_owner()
  : std::logic_error("Invalid service owner.")
{
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_IO_SERVICE_IPP
