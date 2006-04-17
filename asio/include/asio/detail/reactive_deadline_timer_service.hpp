//
// reactive_deadline_timer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_DEADLINE_TIMER_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_DEADLINE_TIMER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <typename Time_Traits, typename Reactor>
class reactive_deadline_timer_service
  : public asio::io_service::service
{
public:
  // The implementation type of the timer. This type is dependent on the
  // underlying implementation of the timer service.
  struct implementation_type
    : private asio::detail::noncopyable
  {
    boost::posix_time::ptime expiry;
    bool might_have_pending_waits;
  };

  // The time type.
  typedef typename Time_Traits::time_type time_type;

  // The duration type.
  typedef typename Time_Traits::duration_type duration_type;

  // Constructor.
  reactive_deadline_timer_service(asio::io_service& io_service)
    : asio::io_service::service(io_service),
      reactor_(asio::use_service<Reactor>(io_service))
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
  }

  // Construct a new timer implementation.
  void construct(implementation_type& impl)
  {
    impl.expiry = boost::posix_time::ptime();
    impl.might_have_pending_waits = false;
  }

  // Destroy a timer implementation.
  void destroy(implementation_type& impl)
  {
    cancel(impl);
  }

  // Cancel any asynchronous wait operations associated with the timer.
  std::size_t cancel(implementation_type& impl)
  {
    if (!impl.might_have_pending_waits)
      return 0;
    std::size_t count = reactor_.cancel_timer(&impl);
    impl.might_have_pending_waits = false;
    return count;
  }

  // Get the expiry time for the timer as an absolute time.
  time_type expires_at(const implementation_type& impl) const
  {
    return Time_Traits::from_utc(impl.expiry);
  }

  // Set the expiry time for the timer as an absolute time.
  std::size_t expires_at(implementation_type& impl,
      const time_type& expiry_time)
  {
    std::size_t count = cancel(impl);
    impl.expiry = Time_Traits::to_utc(expiry_time);
    return count;
  }

  // Get the expiry time for the timer relative to now.
  duration_type expires_from_now(const implementation_type& impl) const
  {
    return Time_Traits::subtract(expires_at(impl), Time_Traits::now());
  }

  // Set the expiry time for the timer relative to now.
  std::size_t expires_from_now(implementation_type& impl,
      const duration_type& expiry_time)
  {
    return expires_at(impl, Time_Traits::add(Time_Traits::now(), expiry_time));
  }

  // Perform a blocking wait on the timer.
  void wait(implementation_type& impl)
  {
    boost::posix_time::ptime now
      = boost::posix_time::microsec_clock::universal_time();
    while (now < impl.expiry)
    {
      boost::posix_time::time_duration timeout = impl.expiry - now;
      ::timeval tv;
      tv.tv_sec = timeout.total_seconds();
      tv.tv_usec = timeout.total_microseconds() % 1000000;
      socket_ops::select(0, 0, 0, 0, &tv);
      now = boost::posix_time::microsec_clock::universal_time();
    }
  }

  template <typename Handler>
  class wait_handler
  {
  public:
    wait_handler(asio::io_service& io_service, Handler handler)
      : io_service_(io_service),
        work_(io_service),
        handler_(handler)
    {
    }

    void operator()(int result)
    {
      asio::error e(result);
      io_service_.post(detail::bind_handler(handler_, e));
    }

  private:
    asio::io_service& io_service_;
    asio::io_service::work work_;
    Handler handler_;
  };

  // Start an asynchronous wait on the timer.
  template <typename Handler>
  void async_wait(implementation_type& impl, Handler handler)
  {
    impl.might_have_pending_waits = true;
    reactor_.schedule_timer(impl.expiry,
        wait_handler<Handler>(owner(), handler), &impl);
  }

private:
  // The selector that performs event demultiplexing for the provider.
  Reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_DEADLINE_TIMER_SERVICE_HPP
