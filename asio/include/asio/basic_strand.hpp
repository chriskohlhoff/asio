//
// basic_strand.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_STRAND_HPP
#define ASIO_BASIC_STRAND_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_io_object.hpp"
#include "asio/strand_service.hpp"
#include "asio/detail/wrapped_handler.hpp"

namespace asio {

/// Provides serialised handler execution.
/**
 * The basic_strand class template provides the ability to post and dispatch
 * handlers with the guarantee that none of those handlers will execute
 * concurrently.
 *
 * Most applications will use the asio::strand typedef.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Safe.
 *
 * @par Concepts:
 * Dispatcher.
 */
template <typename Service = strand_service>
class basic_strand
  : public basic_io_object<Service>
{
public:
  /// Constructor.
  /**
   * Constructs the strand.
   *
   * @param io_service The io_service object that the strand will use to
   * dispatch handlers that are ready to be run.
   */
  explicit basic_strand(asio::io_service& io_service)
    : basic_io_object<Service>(io_service)
  {
  }

  /// Request the strand to invoke the given handler.
  /**
   * This function is used to ask the strand to execute the given handler.
   *
   * The strand object guarantees that only one handler executed through this
   * strand will be invoked at a time. The handler may be executed inside this
   * function if the guarantee can be met.
   *
   * The strand's guarantee is in addition to the guarantee provided by the
   * underlying io_service. The io_service guarantees that the handler will only
   * be called in a thread in which the io_service's run member function is
   * currently being invoked.
   *
   * @param handler The handler to be called. The strand will make a copy of the
   * handler object as required. The function signature of the handler must be:
   * @code void handler(); @endcode
   */
  template <typename Handler>
  void dispatch(Handler handler)
  {
    this->service.dispatch(this->implementation, handler);
  }

  /// Request the strand to invoke the given handler and return
  /// immediately.
  /**
   * This function is used to ask the strand to execute the given handler, but
   * without allowing the strand to call the handler from inside this function.
   *
   * The strand object guarantees that only one handler executed through this
   * strand will be invoked at a time. The strand's guarantee is in addition to
   * the guarantee provided by the underlying io_service. The io_service
   * guarantees that the handler will only be called in a thread in which the
   * io_service's run member function is currently being invoked.
   *
   * @param handler The handler to be called. The strand will make a copy of the
   * handler object as required. The function signature of the handler must be:
   * @code void handler(); @endcode
   */
  template <typename Handler>
  void post(Handler handler)
  {
    this->service.post(this->implementation, handler);
  }

  /// Create a new handler that automatically dispatches the wrapped handler
  /// on the strand.
  /**
   * This function is used to create a new handler function object that, when
   * invoked, will automatically pass the wrapped handler to the strand's
   * dispatch function.
   *
   * @param handler The handler to be wrapped. The strand will make a copy of
   * the handler object as required. The function signature of the handler must
   * be: @code void handler(A1 a1, ... An an); @endcode
   *
   * @return A function object that, when invoked, passes the wrapped handler to
   * the strand's dispatch function. Given a function object with the signature:
   * @code R f(A1 a1, ... An an); @endcode
   * If this function object is passed to the wrap function like so:
   * @code strand.wrap(f); @endcode
   * then the return value is a function object with the signature
   * @code void g(A1 a1, ... An an); @endcode
   * that, when invoked, executes code equivalent to:
   * @code strand.dispatch(boost::bind(f, a1, ... an)); @endcode
   */
  template <typename Handler>
#if defined(GENERATING_DOCUMENTATION)
  unspecified
#else
  detail::wrapped_handler<basic_strand<Service>, Handler>
#endif
  wrap(Handler handler)
  {
    return detail::wrapped_handler<
        basic_strand<Service>,
        Handler>(*this, handler);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_STRAND_HPP
