//
// io_service.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IO_SERVICE_HPP
#define ASIO_IO_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <stdexcept>
#include <typeinfo>
#include <boost/config.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/epoll_reactor_fwd.hpp"
#include "asio/detail/kqueue_reactor_fwd.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/select_reactor_fwd.hpp"
#include "asio/detail/service_registry.hpp"
#include "asio/detail/signal_init.hpp"
#include "asio/detail/task_io_service_fwd.hpp"
#include "asio/detail/win_iocp_io_service_fwd.hpp"
#include "asio/detail/winsock_init.hpp"
#include "asio/detail/wrapped_handler.hpp"

namespace asio {

/// Provides core I/O functionality.
/**
 * The io_service class provides the core I/O functionality for users of the
 * asynchronous I/O objects, including:
 *
 * @li asio::stream_socket
 * @li asio::datagram_socket
 * @li asio::socket_acceptor
 * @li asio::deadline_timer.
 *
 * The io_service class also includes facilities intended for developers of
 * custom asynchronous services.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe, with the exception that calling reset()
 * while there are unfinished run() calls results in undefined behaviour.
 *
 * @par Concepts:
 * Dispatcher.
 *
 * @sa @ref io_service_handler_exception
 */
class io_service
  : private noncopyable
{
private:
  // The type of the platform-specific implementation.
#if defined(ASIO_HAS_IOCP)
  typedef detail::win_iocp_io_service impl_type;
#elif defined(ASIO_HAS_EPOLL)
  typedef detail::task_io_service<detail::epoll_reactor<false> > impl_type;
#elif defined(ASIO_HAS_KQUEUE)
  typedef detail::task_io_service<detail::kqueue_reactor<false> > impl_type;
#else
  typedef detail::task_io_service<detail::select_reactor<false> > impl_type;
#endif

public:
  class work;
  friend class work;

  class service;

  /// Default constructor.
  io_service();

  /// Run the io_service's event processing loop.
  void run();

  /// Interrupt the io_service's event processing loop.
  void interrupt();

  /// Reset the io_service in preparation for a subsequent run() invocation.
  void reset();

  /// Request the io_service to invoke the given handler.
  template <typename Handler>
  void dispatch(Handler handler);

  /// Request the io_service to invoke the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler);

  /// Create a new handler that automatically dispatches the wrapped handler
  /// on the io_service.
  template <typename Handler>
#if defined(GENERATING_DOCUMENTATION)
  unspecified
#else
  detail::wrapped_handler<io_service, Handler>
#endif
  wrap(Handler handler);

  /// Obtain the service object corresponding to the given type.
  template <typename Service>
  friend Service& use_service(io_service& ios);

  /// Add a service object to the io_service.
  template <typename Service>
  friend void add_service(io_service& ios, Service* svc);

  /// Determine if an io_service contains a specified service type.
  template <typename Service>
  friend bool has_service(io_service& ios);

private:
#if defined(BOOST_WINDOWS)
  detail::winsock_init<> init_;
#elif defined(__sun) || defined(__QNX__)
  detail::signal_init<> init_;
#endif

  // The service registry.
  detail::service_registry<io_service> service_registry_;

  // The implementation.
  impl_type& impl_;
};

/// Class to inform the io_service when it has work to do.
/**
 * The work class is used to inform the io_service when work starts and
 * finishes. This ensures that the io_service's run() function will not exit
 * while work is underway, and that it does exit when there is no unfinished
 * work remaining.
 *
 * The work class is copy-constructible so that it may be used as a data member
 * in a handler class. It is not assignable.
 */
class io_service::work
{
public:
  /// Constructor notifies the io_service that work is starting.
  explicit work(io_service& io_service);

  /// Copy constructor notifies the io_service that work is starting.
  work(const work& other);

  /// Destructor notifies the io_service that the work is complete.
  ~work();

private:
  // Prevent assignment.
  void operator=(const work& other);

  // The io_service's implementation.
  io_service::impl_type& impl_;
};

/// Base class for all io_service services.
class io_service::service
  : private noncopyable
{
public:
  /// Get the io_service object that owns the service.
  io_service& owner();

protected:
  /// Constructor.
  service(io_service& owner);

  /// Destructor.
  virtual ~service();

private:
  friend class detail::service_registry<io_service>;
  io_service& owner_;
  const std::type_info* type_info_;
  service* next_;
};

/// Exception thrown when trying to add a duplicate service to an io_service.
class service_already_exists
  : public std::logic_error
{
public:
  service_already_exists()
    : std::logic_error("Service already exists.")
  {
  }
};

/// Exception thrown when trying to add a service object to an io_service where
/// the service has a different owner.
class invalid_service_owner
  : public std::logic_error
{
public:
  invalid_service_owner()
    : std::logic_error("Invalid service owner.")
  {
  }
};

} // namespace asio

#include "asio/impl/io_service.ipp"

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IO_SERVICE_HPP
