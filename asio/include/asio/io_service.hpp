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

#include "asio/service_factory.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/kqueue_reactor.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/service_registry.hpp"
#include "asio/detail/signal_init.hpp"
#include "asio/detail/task_io_service.hpp"
#include "asio/detail/win_iocp_io_service.hpp"
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
 * @sa \ref io_service_handler_exception
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
  /// Default constructor.
  io_service()
    : service_registry_(*this),
      impl_(get_service(service_factory<impl_type>()))
  {
  }

  /// Run the io_service's event processing loop.
  /**
   * The run() function blocks until all work has finished and there are no
   * more handlers to be dispatched, or until the io_service has been
   * interrupted.
   *
   * Multiple threads may call the run() function to set up a pool of threads
   * from which the io_service may execute handlers.
   *
   * The run() function may be safely called again once it has completed only
   * after a call to reset().
   */
  void run()
  {
    impl_.run();
  }

  /// Interrupt the io_service's event processing loop.
  /**
   * This function does not block, but instead simply signals to the io_service
   * that all invocations of its run() member function should return as soon as
   * possible.
   *
   * Note that if the run() function is interrupted and is not called again
   * later then its work may not have finished and handlers may not be
   * delivered. In this case an io_service implementation is not required to
   * make any guarantee that the resources associated with unfinished work will
   * be cleaned up.
   */
  void interrupt()
  {
    impl_.interrupt();
  }

  /// Reset the io_service in preparation for a subsequent run() invocation.
  /**
   * This function must be called prior to any second or later set of
   * invocations of the run() function. It allows the io_service to reset any
   * internal state, such as an interrupt flag.
   *
   * This function must not be called while there are any unfinished calls to
   * the run() function.
   */
  void reset()
  {
    impl_.reset();
  }

  /// Request the io_service to invoke the given handler.
  /**
   * This function is used to ask the io_service to execute the given handler.
   *
   * The io_service guarantees that the handler will only be called in a thread
   * in which the run() member function is currently being invoked. The handler
   * may be executed inside this function if the guarantee can be met.
   *
   * @param handler The handler to be called. The io_service will make
   * a copy of the handler object as required. The function signature of the
   * handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void dispatch(Handler handler)
  {
    impl_.dispatch(handler);
  }

  /// Request the io_service to invoke the given handler and return immediately.
  /**
   * This function is used to ask the io_service to execute the given handler,
   * but without allowing the io_service to call the handler from inside this
   * function.
   *
   * The io_service guarantees that the handler will only be called in a thread
   * in which the run() member function is currently being invoked.
   *
   * @param handler The handler to be called. The io_service will make
   * a copy of the handler object as required. The function signature of the
   * handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void post(Handler handler)
  {
    impl_.post(handler);
  }

  /// Create a new handler that automatically dispatches the wrapped handler
  /// on the io_service.
  /**
   * This function is used to create a new handler function object that, when
   * invoked, will automatically pass the wrapped handler to the io_service's
   * dispatch function.
   *
   * @param handler The handler to be wrapped. The io_service will make a copy
   * of the handler object as required. The function signature of the handler
   * must be: @code void handler(A1 a1, ... An an); @endcode
   *
   * @return A function object that, when invoked, passes the wrapped handler to
   * the io_service's dispatch function. Given a function object with the
   * signature:
   * @code R f(A1 a1, ... An an); @endcode
   * If this function object is passed to the wrap function like so:
   * @code io_service.wrap(f); @endcode
   * then the return value is a function object with the signature
   * @code void g(A1 a1, ... An an); @endcode
   * that, when invoked, executes code equivalent to:
   * @code io_service.dispatch(boost::bind(f, a1, ... an)); @endcode
   */
  template <typename Handler>
#if defined(GENERATING_DOCUMENTATION)
  unspecified
#else
  detail::wrapped_handler<io_service, Handler>
#endif
  wrap(Handler handler)
  {
    return detail::wrapped_handler<io_service, Handler>(*this, handler);
  }

  /// Obtain the service interface corresponding to the given type.
  /**
   * This function is used to locate a service interface that corresponds to
   * the given service type. If there is no existing implementation of the
   * service, then the io_service will use the supplied factory to create a new
   * instance.
   *
   * @param factory The factory to use to create the service.
   *
   * @return The service interface implementing the specified service type.
   * Ownership of the service interface is not transferred to the caller.
   */
  template <typename Service>
  Service& get_service(service_factory<Service> factory)
  {
    return service_registry_.get_service(factory);
  }

  class work;
  friend class work;

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
  /**
   * The constructor is used to inform the io_service that some work has begun.
   * This ensures that the io_service's run() function will not exit while the
   * work is underway.
   */
  explicit work(io_service& io_service)
    : impl_(io_service.impl_)
  {
    impl_.work_started();
  }

  /// Copy constructor notifies the io_service that work is starting.
  /**
   * The constructor is used to inform the io_service that some work has begun.
   * This ensures that the io_service's run() function will not exit while the
   * work is underway.
   */
  work(const work& other)
    : impl_(other.impl_)
  {
    impl_.work_started();
  }

  /// Destructor notifies the io_service that the work is complete.
  /**
   * The destructor is used to inform the io_service that some work has
   * finished. Once the count of unfinished work reaches zero, the io_service's
   * run() function is permitted to exit.
   */
  ~work()
  {
    impl_.work_finished();
  }

private:
  // Prevent assignment.
  void operator=(const work& other);

  // The io_service's implementation.
  io_service::impl_type& impl_;
};

/**
 * @page io_service_handler_exception Effect of exceptions thrown from handlers
 *
 * If an exception is thrown from a handler, the exception is allowed to
 * propagate through the throwing thread's invocation of
 * asio::io_service::run(). No other threads that are calling
 * asio::io_service::run() are affected. It is then the responsibility of
 * the application to catch the exception.
 *
 * After the exception has been caught, the asio::io_service::run() call
 * may be restarted @em without the need for an intervening call to
 * asio::io_service::reset(). This allows the thread to rejoin the
 * io_service's thread pool without impacting any other threads in the
 * pool.
 *
 * @par Example:
 * @code
 * asio::io_service io_service;
 * ...
 * for (;;)
 * {
 *   try
 *   {
 *     io_service.run();
 *     break; // run() exited normally
 *   }
 *   catch (my_exception& e)
 *   {
 *     // Deal with exception as appropriate.
 *   }
 * }
 * @endcode
 */

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IO_SERVICE_HPP
