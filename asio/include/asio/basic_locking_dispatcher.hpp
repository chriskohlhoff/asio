//
// basic_locking_dispatcher.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

#include "asio/detail/push_options.hpp"
#include <memory>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/locking_dispatcher.hpp"
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
 * Dispatcher.
 */
template <typename Dispatcher, typename Allocator = std::allocator<void> >
class basic_locking_dispatcher
  : private noncopyable
{
public:
  /// The underlying dispatcher type.
  typedef Dispatcher dispatcher_type;

  /// The allocator type for the locking dispatcher.
  typedef Allocator allocator_type;

  /// Constructor.
  /**
   * Constructs the locking dispatcher.
   *
   * @param dispatcher The dispatcher object that the locking_dispatcher will
   * use to dispatch handlers that are ready to be run.
   *
   * @param allocator The allocator object used by the locking_dispatcher to
   * dynamically allocate internal objects.
   */
  explicit basic_locking_dispatcher(dispatcher_type& dispatcher,
      const allocator_type& allocator = allocator_type())
    : impl_(dispatcher, allocator)
  {
  }

  /// Destructor.
  ~basic_locking_dispatcher()
  {
  }

  /// Get the underlying dispatcher associated with the locking_dispatcher.
  /**
   * This function may be used to obtain the dispatcher object that the locking
   * dispatcher uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the dispatcher object that the locking dispatcher
   * will use to dispatch handlers. Ownership is not transferred to the caller.
   */
  dispatcher_type& dispatcher()
  {
    return impl_.dispatcher();
  }

  /// Return a copy of the allocator associated with the locking_dispatcher.
  /**
   * The get_allocator() returns a copy of the allocator object used by the
   * locking_dispatcher.
   *
   * @return A copy of the locking_dispatcher's allocator.
   */
  allocator_type get_allocator() const
  {
    return impl_.get_allocator();
  }

  /// Request the dispatcher to invoke the given handler.
  /**
   * This function is used to ask the dispatcher to execute the given handler.
   *
   * The locking dispatcher guarantees that only one handler executed through
   * this dispatcher will be invoked at a time. The handler may be executed
   * inside this function if the guarantee can be met.
   *
   * The locking dispatcher's guarantee is in addition to any guarantee
   * provided by the underlying dispatcher. For example, the io_service
   * guarantees that the handler will only be called in a thread in which the
   * io_services's run member function is currently being invoked.
   *
   * @param handler The handler to be called. The locking dispatcher will make
   * a copy of the handler object as required. The function signature of the
   * handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void dispatch(Handler handler)
  {
    impl_.dispatch(handler);
  }

  /// Request the dispatcher to invoke the given handler and return
  /// immediately.
  /**
   * This function is used to ask the dispatcher to execute the given handler,
   * but without allowing the dispatcher to call the handler from inside this
   * function.
   *
   * The locking dispatcher guarantees that only one handler executed through
   * this dispatcher will be invoked at a time. The handler may be executed
   * inside this function if the guarantee can be met.
   *
   * The locking dispatcher's guarantee is in addition to any guarantee
   * provided by the underlying dispatcher. For example, the io_service
   * guarantees that the handler will only be called in a thread in which the
   * io_services's run member function is currently being invoked.
   *
   * @param handler The handler to be called. The locking dispatcher will make
   * a copy of the handler object as required. The function signature of the
   * handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void post(Handler handler)
  {
    impl_.post(handler);
  }

  /// Create a new handler that automatically dispatches the wrapped handler
  /// on the dispatcher.
  /**
   * This function is used to create a new handler function object that, when
   * invoked, will automatically pass the wrapped handler to the dispatcher's
   * dispatch function.
   *
   * @param handler The handler to be wrapped. The dispatcher will make a copy
   * of the handler object as required. The function signature of the handler
   * must be: @code void handler(A1 a1, ... An an); @endcode
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
  detail::wrapped_handler<
      basic_locking_dispatcher<Dispatcher, Allocator>,
      Handler>
#endif
  wrap(Handler handler)
  {
    return detail::wrapped_handler<
        basic_locking_dispatcher<Dispatcher, Allocator>,
        Handler>(*this, handler);
  }

private:
  /// The underlying native implementation.
  detail::locking_dispatcher<Dispatcher, Allocator> impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_LOCKING_DISPATCHER_HPP
