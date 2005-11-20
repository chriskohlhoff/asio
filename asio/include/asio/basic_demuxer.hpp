//
// basic_demuxer.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_DEMUXER_HPP
#define ASIO_BASIC_DEMUXER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/service_factory.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/service_registry.hpp"
#include "asio/detail/signal_init.hpp"
#include "asio/detail/winsock_init.hpp"
#include "asio/detail/wrapped_handler.hpp"

namespace asio {

/// Provides core event demultiplexing functionality.
/**
 * The basic_demuxer class template provides the core event demultiplexing
 * functionality for users of the asynchronous I/O objects, including:
 *
 * @li asio::stream_socket
 * @li asio::datagram_socket
 * @li asio::socket_acceptor
 * @li asio::deadline_timer.
 *
 * The basic_demuxer class template also includes facilities intended for
 * developers of custom asynchronous services.
 *
 * Most applications will use the asio::demuxer typedef.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe, with the exception that calling reset()
 * while there are unfinished run() calls results in undefined behaviour.
 *
 * @par Concepts:
 * Dispatcher.
 */
template <typename Demuxer_Service>
class basic_demuxer
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide demuxer operations.
  typedef Demuxer_Service service_type;

  /// Default constructor.
  basic_demuxer()
    : service_registry_(*this),
      service_(get_service(service_factory<Demuxer_Service>()))
  {
  }

  /// Construct using the supplied service_factory to get the demuxer service.
  explicit basic_demuxer(const service_factory<Demuxer_Service>& factory)
    : service_registry_(*this),
      service_(get_service(factory))
  {
  }

  /// Run the demuxer's event processing loop.
  /**
   * The run() function blocks until all work has finished and there are no
   * more handlers to be dispatched, or until the demuxer has been interrupted.
   *
   * Multiple threads may call the run() function to set up a pool of threads
   * from which the demuxer may execute handlers.
   *
   * The run() function may be safely called again once it has completed only
   * after a call to reset().
   */
  void run()
  {
    service_.run();
  }

  /// Interrupt the demuxer's event processing loop.
  /**
   * This function does not block, but instead simply signals to the demuxer
   * that all invocations of its run() member function should return as soon as
   * possible.
   *
   * Note that if the run() function is interrupted and is not called again
   * later then its work may not have finished and handlers may not be
   * delivered. In this case a demuxer implementation is not required to make
   * any guarantee that the resources associated with unfinished work will be
   * cleaned up.
   */
  void interrupt()
  {
    service_.interrupt();
  }

  /// Reset the demuxer in preparation for a subsequent run() invocation.
  /**
   * This function must be called prior to any second or later set of
   * invocations of the run() function. It allows the demuxer to reset any
   * internal state, such as an interrupt flag.
   *
   * This function must not be called while there are any unfinished calls to
   * the run() function.
   */
  void reset()
  {
    service_.reset();
  }

  /// Request the demuxer to invoke the given handler.
  /**
   * This function is used to ask the demuxer to execute the given handler.
   *
   * The demuxer guarantees that the handler will only be called in a thread in
   * which the run() member function is currently being invoked. The handler
   * may be executed inside this function if the guarantee can be met.
   *
   * @param handler The handler to be called. The demuxer will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void dispatch(Handler handler)
  {
    service_.dispatch(handler);
  }

  /// Request the demuxer to invoke the given handler and return immediately.
  /**
   * This function is used to ask the demuxer to execute the given handler, but
   * without allowing the demuxer to call the handler from inside this
   * function.
   *
   * The demuxer guarantees that the handler will only be called in a thread in
   * which the run() member function is currently being invoked.
   *
   * @param handler The handler to be called. The demuxer will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void post(Handler handler)
  {
    service_.post(handler);
  }

  /// Create a new handler that automatically dispatches the wrapped handler
  /// on the demuxer.
  /**
   * This function is used to create a new handler function object that, when
   * invoked, will automatically pass the wrapped handler to the demuxer's
   * dispatch function.
   *
   * @param handler The handler to be wrapped. The demuxer will make a copy of
   * the handler object as required. The equivalent function signature of the
   * handler must be: @code void handler(A1 a1, ... An an); @endcode
   *
   * @return A function object that, when invoked, passes the wrapped handler to
   * the demuxer's dispatch function. Given a function object with the
   * signature:
   * @code R f(A1 a1, ... An an); @endcode
   * If this function object is passed to the wrap function like so:
   * @code demuxer.wrap(f); @endcode
   * then the return value is a function object with the signature
   * @code void g(A1 a1, ... An an); @endcode
   * that, when invoked, executes code equivalent to:
   * @code demuxer.dispatch(boost::bind(f, a1, ... an)); @endcode
   */
  template <typename Handler>
#if defined(GENERATING_DOCUMENTATION)
  unspecified
#else
  detail::wrapped_handler<basic_demuxer<Demuxer_Service>, Handler>
#endif
  wrap(Handler handler)
  {
    return detail::wrapped_handler<basic_demuxer<Demuxer_Service>, Handler>(
        *this, handler);
  }

  /// Obtain the service interface corresponding to the given type.
  /**
   * This function is used to locate a service interface that corresponds to
   * the given service type. If there is no existing implementation of the
   * service, then the demuxer will use the supplied factory to create a new
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
#else
  detail::signal_init<> init_;
#endif

  /// The service registry.
  detail::service_registry<basic_demuxer<Demuxer_Service> > service_registry_;

  /// The underlying demuxer service implementation.
  Demuxer_Service& service_;
};

/// Class to inform the demuxer when it has work to do.
/**
 * The work class is used to inform the demuxer when work starts and finishes.
 * This ensures that the demuxer's run() function will not exit while work is
 * underway, and that it does exit when there is no unfinished work remaining.
 *
 * The work class is copy-constructible so that it may be used as a data member
 * in a handler class. It is not assignable.
 */
template <typename Demuxer_Service>
class basic_demuxer<Demuxer_Service>::work
{
public:
  /// Constructor notifies the demuxer that work is starting.
  /**
   * The constructor is used to inform the demuxer that some work has begun.
   * This ensures that the demuxer's run() function will not exit while the work
   * is underway.
   */
  explicit work(basic_demuxer<Demuxer_Service>& demuxer)
    : service_(demuxer.service_)
  {
    service_.work_started();
  }

  /// Copy constructor notifies the demuxer that work is starting.
  /**
   * The constructor is used to inform the demuxer that some work has begun.
   * This ensures that the demuxer's run() function will not exit while the work
   * is underway.
   */
  work(const work& other)
    : service_(other.service_)
  {
    service_.work_started();
  }

  /// Destructor notifies the demuxer that the work is complete.
  /**
   * The destructor is used to inform the demuxer that some work has finished.
   * Once the count of unfinished work reaches zero, the demuxer's run()
   * function is permitted to exit.
   */
  ~work()
  {
    service_.work_finished();
  }

private:
  // Prevent assignment.
  void operator=(const work& other);

  /// The underlying demuxer service implementation.
  Demuxer_Service& service_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_DEMUXER_HPP
