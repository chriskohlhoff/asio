//
// demuxer_service.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DEMUXER_SERVICE_HPP
#define ASIO_DEMUXER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <memory>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/service_factory.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/kqueue_reactor.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/task_demuxer_service.hpp"
#include "asio/detail/win_iocp_demuxer_service.hpp"

namespace asio {

/// Default service implementation for a demuxer.
template <typename Allocator = std::allocator<void> >
class demuxer_service
  : private noncopyable
{
public:
  /// The demuxer type for this service.
  typedef basic_demuxer<demuxer_service<Allocator>, Allocator> demuxer_type;

private:
  // The type of the platform-specific implementation.
#if defined(ASIO_HAS_IOCP_DEMUXER)
  typedef detail::win_iocp_demuxer_service<Allocator> service_impl_type;
#elif defined(ASIO_HAS_EPOLL_REACTOR)
  typedef detail::task_demuxer_service<detail::epoll_reactor<false> >
    service_impl_type;
#elif defined(ASIO_HAS_KQUEUE_REACTOR)
  typedef detail::task_demuxer_service<detail::kqueue_reactor<false> >
    service_impl_type;
#else
  typedef detail::task_demuxer_service<detail::select_reactor<false> >
    service_impl_type;
#endif

public:
  /// Constructor.
  explicit demuxer_service(demuxer_type& demuxer)
    : service_impl_(demuxer.get_service(service_factory<service_impl_type>()))
  {
  }

  /// Run the demuxer's event processing loop.
  void run()
  {
    service_impl_.run();
  }

  /// Interrupt the demuxer's event processing loop.
  void interrupt()
  {
    service_impl_.interrupt();
  }

  /// Reset the demuxer in preparation for a subsequent run invocation.
  void reset()
  {
    service_impl_.reset();
  }

  /// Notify the demuxer that some work has started.
  void work_started()
  {
    service_impl_.work_started();
  }

  /// Notify the demuxer that some work has finished.
  void work_finished()
  {
    service_impl_.work_finished();
  }

  /// Request the demuxer to invoke the given handler.
  template <typename Handler>
  void dispatch(Handler handler)
  {
    service_impl_.dispatch(handler);
  }

  /// Request the demuxer to invoke the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler)
  {
    service_impl_.post(handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DEMUXER_SERVICE_HPP
