//
// basic_locking_dispatcher.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_LOCKING_DISPATCHER_HPP
#define ASIO_BASIC_LOCKING_DISPATCHER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/wrapped_handler.hpp"

namespace asio {

/// Provides serialised handler execution.
/**
 * The basic_locking_dispatcher class template provides the ability to post
 * and dispatch handlers with the guarantee that none of those handlers will
 * execute concurrently.
 *
 * Most applications will use the asio::locking_dispatcher typedef.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe.
 *
 * @par Concepts:
 * Async_Object, Dispatcher, Error_Source.
 */
template <typename Service>
class basic_locking_dispatcher
  : private noncopyable
{
public:
  /// The type of the service that will be used to provide locking dispatcher
  /// operations.
  typedef Service service_type;

  /// The native implementation type of the locking dispatcher.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this dispatcher.
  typedef typename service_type::demuxer_type demuxer_type;

  /// The type used for reporting errors.
  typedef asio::error error_type;

  /// Constructor.
  /**
   * Constructs the locking dispatcher.
   *
   * @param d The demuxer object that the locking dispatcher will use to
   * dispatch handlers that are ready to be run.
   */
  explicit basic_locking_dispatcher(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
  }

  /// Destructor.
  ~basic_locking_dispatcher()
  {
    service_.destroy(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the locking
   * dispatcher uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the demuxer object that the dispatcher will use to
   * dispatch handlers. Ownership is not transferred to the caller.
   */
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Request the dispatcher to invoke the given handler.
  /**
   * This function is used to ask the dispatcher to execute the given handler.
   *
   * The dispatcher guarantees that the handler will only be called in a thread
   * in which the underlying demuxer's run member function is currently being
   * invoked. It also guarantees that only one handler executed through this
   * dispatcher will be invoked at a time. The handler may be executed inside
   * this function if the guarantee can be met.
   *
   * @param handler The handler to be called. The dispatcher will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void dispatch(Handler handler)
  {
    service_.dispatch(impl_, handler);
  }

  /// Request the dispatcher to invoke the given handler and return
  /// immediately.
  /**
   * This function is used to ask the dispatcher to execute the given handler,
   * but without allowing the dispatcher to call the handler from inside this
   * function.
   *
   * The dispatcher guarantees that the handler will only be called in a thread
   * in which the underlying demuxer's run member function is currently being
   * invoked. It also guarantees that only one handler executed through this
   * dispatcher will be invoked at a time.
   *
   * @param handler The handler to be called. The dispatcher will make
   * a copy of the handler object as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void post(Handler handler)
  {
    service_.post(impl_, handler);
  }

  /// Create a new handler that automatically dispatches the wrapped handler
  /// on the dispatcher.
  /**
   * This function is used to create a new handler function object that, when
   * invoked, will automatically pass the wrapped handler to the dispatcher's
   * dispatch function.
   *
   * @param handler The handler to be wrapped. The dispatcher will make a copy
   * of the handler object as required. The equivalent function signature of
   * the handler must be: @code void handler(A1 a1, ... An an); @endcode
   *
   * @return A function object that, when invoked, passes the wrapped handler to
   * the dispatcher's dispatch function. Given a function object with the
   * signature:
   * @code R f(A1 a1, ... An an); @endcode
   * If this function object is passed to the wrap function like so:
   * @code dispatcher.wrap(f); @endcode
   * then the return value is a function object with the signature
   * @code void g(A1 a1, ... An an); @endcode
   * that, when invoked, executes code equivalent to:
   * @code dispatcher.dispatch(boost::bind(f, a1, ... an)); @endcode
   */
  template <typename Handler>
#if defined(GENERATING_DOCUMENTATION)
  unspecified
#else
  detail::wrapped_handler<basic_locking_dispatcher<Service>, Handler>
#endif
  wrap(Handler handler)
  {
    return detail::wrapped_handler<basic_locking_dispatcher<Service>, Handler>(
        *this, handler);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_LOCKING_DISPATCHER_HPP
