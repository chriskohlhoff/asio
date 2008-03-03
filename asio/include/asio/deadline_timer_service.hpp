//
// deadline_timer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DEADLINE_TIMER_SERVICE_HPP
#define ASIO_DEADLINE_TIMER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/io_service.hpp"
#include "asio/time_traits.hpp"
#include "asio/detail/deadline_timer_service.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/kqueue_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/service_base.hpp"
#include "asio/detail/win_iocp_io_service.hpp"

namespace asio {

/// Default service implementation for a timer.
template <typename TimeType,
    typename TimeTraits = asio::time_traits<TimeType> >
class deadline_timer_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_service::service
#else
  : public asio::detail::service_base<
      deadline_timer_service<TimeType, TimeTraits> >
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_service::id id;
#endif

  /// The time traits type.
  typedef TimeTraits traits_type;

  /// The time type.
  typedef typename traits_type::time_type time_type;

  /// The duration type.
  typedef typename traits_type::duration_type duration_type;

private:
  // The type of the platform-specific implementation.
#if defined(ASIO_HAS_IOCP)
  typedef detail::deadline_timer_service<
    traits_type, detail::win_iocp_io_service> service_impl_type;
#elif defined(ASIO_HAS_EPOLL)
  typedef detail::deadline_timer_service<
    traits_type, detail::epoll_reactor<false> > service_impl_type;
#elif defined(ASIO_HAS_KQUEUE)
  typedef detail::deadline_timer_service<
    traits_type, detail::kqueue_reactor<false> > service_impl_type;
#elif defined(ASIO_HAS_DEV_POLL)
  typedef detail::deadline_timer_service<
    traits_type, detail::dev_poll_reactor<false> > service_impl_type;
#else
  typedef detail::deadline_timer_service<
    traits_type, detail::select_reactor<false> > service_impl_type;
#endif

public:
  /// The implementation type of the deadline timer.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef typename service_impl_type::implementation_type implementation_type;
#endif

  /// Construct a new timer service for the specified io_service.
  explicit deadline_timer_service(asio::io_service& io_service)
    : asio::detail::service_base<
        deadline_timer_service<TimeType, TimeTraits> >(io_service),
      service_impl_(asio::use_service<service_impl_type>(io_service))
  {
  }

  /// Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
  }

  /// Construct a new timer implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a timer implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Cancel any asynchronous wait operations associated with the timer.
  std::size_t cancel(implementation_type& impl, asio::error_code& ec)
  {
    return service_impl_.cancel(impl, ec);
  }

  /// Get the expiry time for the timer as an absolute time.
  time_type expires_at(const implementation_type& impl) const
  {
    return service_impl_.expires_at(impl);
  }

  /// Set the expiry time for the timer as an absolute time.
  std::size_t expires_at(implementation_type& impl,
      const time_type& expiry_time, asio::error_code& ec)
  {
    return service_impl_.expires_at(impl, expiry_time, ec);
  }

  /// Get the expiry time for the timer relative to now.
  duration_type expires_from_now(const implementation_type& impl) const
  {
    return service_impl_.expires_from_now(impl);
  }

  /// Set the expiry time for the timer relative to now.
  std::size_t expires_from_now(implementation_type& impl,
      const duration_type& expiry_time, asio::error_code& ec)
  {
    return service_impl_.expires_from_now(impl, expiry_time, ec);
  }

  // Perform a blocking wait on the timer.
  void wait(implementation_type& impl, asio::error_code& ec)
  {
    service_impl_.wait(impl, ec);
  }

  // Start an asynchronous wait on the timer.
  template <typename WaitHandler>
  void async_wait(implementation_type& impl, WaitHandler handler)
  {
    service_impl_.async_wait(impl, handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DEADLINE_TIMER_SERVICE_HPP
