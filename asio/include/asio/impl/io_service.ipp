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
inline void io_service::run()
{
  impl_.run();
}

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
inline void io_service::interrupt()
{
  impl_.interrupt();
}

/**
 * This function must be called prior to any second or later set of
 * invocations of the run() function. It allows the io_service to reset any
 * internal state, such as an interrupt flag.
 *
 * This function must not be called while there are any unfinished calls to
 * the run() function.
 */
inline void io_service::reset()
{
  impl_.reset();
}

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
inline void io_service::dispatch(Handler handler)
{
  impl_.dispatch(handler);
}

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
inline void io_service::post(Handler handler)
{
  impl_.post(handler);
}

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
inline detail::wrapped_handler<io_service, Handler>
#endif
io_service::wrap(Handler handler)
{
  return detail::wrapped_handler<io_service, Handler>(*this, handler);
}

/**
 * The constructor is used to inform the io_service that some work has begun.
 * This ensures that the io_service's run() function will not exit while the
 * work is underway.
 */
inline io_service::work::work(io_service& io_service)
  : impl_(io_service.impl_)
{
  impl_.work_started();
}

/**
 * The constructor is used to inform the io_service that some work has begun.
 * This ensures that the io_service's run() function will not exit while the
 * work is underway.
 */
inline io_service::work::work(const work& other)
  : impl_(other.impl_)
{
  impl_.work_started();
}

/**
 * The destructor is used to inform the io_service that some work has
 * finished. Once the count of unfinished work reaches zero, the io_service's
 * run() function is permitted to exit.
 */
inline io_service::work::~work()
{
  impl_.work_finished();
}

/**
 * @param owner The io_service object that owns the service.
 */
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

/**
 * This function is used to locate a service object that corresponds to
 * the given service type. If there is no existing implementation of the
 * service, then the io_service will create a new instance of the service.
 *
 * @param ios The io_service object that owns the service.
 *
 * @return The service interface implementing the specified service type.
 * Ownership of the service interface is not transferred to the caller.
 */
template <typename Service>
inline Service& use_service(io_service& ios)
{
  return ios.service_registry_.template use_service<Service>();
}

/**
 * This function is used to add a service to the io_service.
 *
 * @param ios The io_service object that owns the service.
 *
 * @param svc The service object. On success, ownership of the service object is
 * transferred to the io_service. When the io_service object is destroyed, it
 * will destroy the service object by performing:
 * @code delete static_cast<io_service::service*>(svc) @endcode
 *
 * @throws asio::service_already_exists Thrown if a service of the given
 * type is already present in the io_service.
 *
 * @throws asio::invalid_service_owner Thrown if the service's owning
 * io_service is not the io_service object specified by the ios parameter.
 */
template <typename Service>
void add_service(io_service& ios, Service* svc)
{
  if (&ios != &svc->owner())
    boost::throw_exception(invalid_service_owner());
  if (!ios.service_registry_.template add_service<Service>(svc))
    boost::throw_exception(service_already_exists());
}

/**
 * This function is used to determine whether the io_service contains a service
 * object corresponding to the given service type.
 *
 * @param ios The io_service object that owns the service.
 *
 * @return A boolean indicating whether the io_service contains the service.
 */
template <typename Service>
bool has_service(io_service& ios)
{
  return ios.service_registry_.template has_service<Service>();
}

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

#endif // ASIO_IO_SERVICE_IPP
