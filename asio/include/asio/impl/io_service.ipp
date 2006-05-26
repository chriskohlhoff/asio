//
// io_service.ipp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IO_SERVICE_IPP
#define ASIO_IO_SERVICE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/kqueue_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/task_io_service.hpp"
#include "asio/detail/win_iocp_io_service.hpp"

namespace asio {

inline io_service::io_service()
  : service_registry_(*this),
    impl_(service_registry_.use_service<impl_type>())
{
}

inline void io_service::run()
{
  impl_.run();
}

inline void io_service::interrupt()
{
  impl_.interrupt();
}

inline void io_service::reset()
{
  impl_.reset();
}

template <typename Handler>
inline void io_service::dispatch(Handler handler)
{
  impl_.dispatch(handler);
}

template <typename Handler>
inline void io_service::post(Handler handler)
{
  impl_.post(handler);
}

template <typename Handler>
#if defined(GENERATING_DOCUMENTATION)
unspecified
#else
inline detail::wrapped_handler<io_service, Handler>
#endif
io_service::wrap(Handler handler)
{
  return detail::wrapped_handler<io_service, Handler>(*this, handler);
}

inline io_service::work::work(io_service& io_service)
  : impl_(io_service.impl_)
{
  impl_.work_started();
}

inline io_service::work::work(const work& other)
  : impl_(other.impl_)
{
  impl_.work_started();
}

inline io_service::work::~work()
{
  impl_.work_finished();
}

inline io_service::service::service(io_service& owner)
  : owner_(owner),
    type_info_(0),
    next_(0)
{
}

inline io_service::service::~service()
{
}

inline io_service& io_service::service::owner()
{
  return owner_;
}

template <typename Service>
inline Service& use_service(io_service& ios)
{
  return ios.service_registry_.template use_service<Service>();
}

template <typename Service>
void add_service(io_service& ios, Service* svc)
{
  if (&ios != &svc->owner())
    boost::throw_exception(invalid_service_owner());
  if (!ios.service_registry_.template add_service<Service>(svc))
    boost::throw_exception(service_already_exists());
}

template <typename Service>
bool has_service(io_service& ios)
{
  return ios.service_registry_.template has_service<Service>();
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IO_SERVICE_IPP
