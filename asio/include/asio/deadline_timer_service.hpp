//
// deadline_timer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
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

#include "asio/detail/socket_types.hpp" // Must come before posix_time.

#include "asio/detail/push_options.hpp"
#include <memory>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/time_traits.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/kqueue_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_deadline_timer_service.hpp"
#include "asio/detail/win_iocp_demuxer_service.hpp"

namespace asio {

/// Default service implementation for a timer.
template <typename Time_Type = boost::posix_time::ptime,
    typename Time_Traits = asio::time_traits<Time_Type>,
    typename Allocator = std::allocator<void> >
class deadline_timer_service
  : private boost::noncopyable
{
public:
  /// The demuxer type.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

  /// The time traits type.
  typedef Time_Traits traits_type;

  /// The time type.
  typedef typename traits_type::time_type time_type;

  /// The duration type.
  typedef typename traits_type::duration_type duration_type;

private:
  // The type of the platform-specific implementation.
#if defined(ASIO_HAS_IOCP_DEMUXER)
  typedef detail::reactive_deadline_timer_service<demuxer_type,
    traits_type, detail::select_reactor<true> > service_impl_type;
#elif defined(ASIO_HAS_EPOLL_REACTOR)
  typedef detail::reactive_deadline_timer_service<demuxer_type,
    traits_type, detail::epoll_reactor<false> > service_impl_type;
#elif defined(ASIO_HAS_KQUEUE_REACTOR)
  typedef detail::reactive_deadline_timer_service<demuxer_type,
    traits_type, detail::kqueue_reactor<false> > service_impl_type;
#else
  typedef detail::reactive_deadline_timer_service<demuxer_type,
    traits_type, detail::select_reactor<false> > service_impl_type;
#endif

public:
  /// The native type of the socket acceptor.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined impl_type;
#else
  typedef typename service_impl_type::impl_type impl_type;
#endif

  /// Construct a new timer service for the specified demuxer.
  explicit deadline_timer_service(demuxer_type& demuxer)
    : service_impl_(demuxer.get_service(service_factory<service_impl_type>()))
  {
  }

  /// Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return service_impl_.demuxer();
  }

  /// Return a null socket acceptor implementation.
  impl_type null() const
  {
    return service_impl_.null();
  }

  /// Create a new timer implementation.
  void create(impl_type& impl)
  {
    service_impl_.create(impl);
  }

  /// Destroy a timer implementation.
  void destroy(impl_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Get the expiry time for the timer as an absolute time.
  time_type expires_at(const impl_type& impl) const
  {
    return service_impl_.expires_at(impl);
  }

  /// Set the expiry time for the timer as an absolute time.
  void expires_at(impl_type& impl, const time_type& expiry_time)
  {
    service_impl_.expires_at(impl, expiry_time);
  }

  /// Get the expiry time for the timer relative to now.
  duration_type expires_from_now(const impl_type& impl) const
  {
    return service_impl_.expires_from_now(impl);
  }

  /// Set the expiry time for the timer relative to now.
  void expires_from_now(impl_type& impl, const duration_type& expiry_time)
  {
    service_impl_.expires_from_now(impl, expiry_time);
  }

  /// Cancel any asynchronous wait operations associated with the timer.
  int cancel(impl_type& impl)
  {
    return service_impl_.cancel(impl);
  }

  // Perform a blocking wait on the timer.
  void wait(impl_type& impl)
  {
    service_impl_.wait(impl);
  }

  // Start an asynchronous wait on the timer.
  template <typename Handler>
  void async_wait(impl_type& impl, Handler handler)
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
