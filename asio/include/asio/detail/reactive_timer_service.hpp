//
// reactive_timer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_REACTIVE_TIMER_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_TIMER_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/timer_base.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/time.hpp"

namespace asio {
namespace detail {

template <typename Demuxer, typename Reactor>
class reactive_timer_service
{
public:
  // Implementation structure for a timer.
  struct timer_impl
    : private boost::noncopyable
  {
    time expiry;
  };

  // The native type of the timer. This type is dependent on the underlying
  // implementation of the timer service.
  typedef timer_impl* impl_type;

  // Return a null timer implementation.
  static impl_type null()
  {
    return 0;
  }

  // Constructor.
  reactive_timer_service(Demuxer& d)
    : demuxer_(d),
      reactor_(d.get_service(service_factory<Reactor>()))
  {
  }

  // The demuxer type for this service.
  typedef Demuxer demuxer_type;

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Create a new timer implementation.
  void create(impl_type& impl)
  {
    impl = new timer_impl;
  }

  // Destroy a timer implementation.
  void destroy(impl_type& impl)
  {
    if (impl != null())
    {
      reactor_.cancel_timer(impl);
      delete impl;
      impl = null();
    }
  }

  // Set the timer.
  void set(impl_type& impl, timer_base::from_type from_when, long sec,
      long usec)
  {
    time now = time::now();
    time relative_time(sec, usec);
    switch (from_when)
    {
    case timer_base::from_now:
      impl->expiry = now;
      impl->expiry += relative_time;
      break;
    case timer_base::from_existing:
      impl->expiry += relative_time;
      break;
    case timer_base::from_epoch:
    default:
      impl->expiry = relative_time;
      break;
    }
  }

  // Cancel any asynchronous wait operations associated with the timer.
  int cancel(impl_type& impl)
  {
    return reactor_.cancel_timer(impl);
  }

  // Perform a blocking wait on the timer.
  void wait(impl_type& impl)
  {
    time now = time::now();
    while (now < impl->expiry)
    {
      time timeout = impl->expiry;
      timeout -= now;
      ::timeval tv;
      tv.tv_sec = timeout.sec();
      tv.tv_usec = timeout.usec();
      socket_ops::select(0, 0, 0, 0, &tv);
      now = time::now();
    }
  }

  template <typename Handler>
  class wait_handler
  {
  public:
    wait_handler(impl_type& impl, Demuxer& demuxer, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        handler_(handler)
    {
    }

    void do_operation()
    {
      asio::error e(asio::error::success);
      demuxer_.post(detail::bind_handler(handler_, e));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error e(asio::error::operation_aborted);
      demuxer_.post(detail::bind_handler(handler_, e));
      demuxer_.work_finished();
    }

  private:
    impl_type& impl_;
    Demuxer& demuxer_;
    Handler handler_;
  };

  // Start an asynchronous wait on the timer.
  template <typename Handler>
  void async_wait(impl_type& impl, Handler handler)
  {
    demuxer_.work_started();
    reactor_.schedule_timer(impl->expiry.sec(), impl->expiry.usec(),
        wait_handler<Handler>(impl, demuxer_, handler), impl);
  }

private:
  // The demuxer used for dispatching handlers.
  Demuxer& demuxer_;

  // The selector that performs event demultiplexing for the provider.
  Reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_TIMER_SERVICE_HPP
