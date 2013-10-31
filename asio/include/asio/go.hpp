//
// go.hpp
// ~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_GO_HPP
#define ASIO_GO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/coroutine.hpp"
#include "asio/detail/wrapped_handler.hpp"
#include "asio/io_service.hpp"
#include "asio/strand.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {
  
template <typename Handler> class stackless_impl_base;

} // namespace detail

/// Context object the represents the currently executing coroutine.
/**
 * The basic_stackless_context class is used to represent the currently
 * executing stackless coroutine. A basic_stackless_context may be passed as a
 * handler to an asynchronous operation. For example:
 *
 * @code template <typename Handler>
 * void my_coroutine(basic_stackless_context<Handler> ctx)
 * {
 *   reenter (ctx)
 *   {
 *     ...
 *     yield my_socket.async_read_some(buffer, ctx);
 *     ...
 *   }
 * } @endcode
 *
 * The initiating function (async_read_some in the above example) suspends the
 * current coroutine. The coroutine is resumed when the asynchronous operation
 * completes.
 */
template <typename Handler>
class basic_stackless_context
{
public:
  /// Return a coroutine context that sets the specified error_code.
  /**
   * By default, when a coroutine context is used with an asynchronous operation, a
   * non-success error_code is converted to system_error and thrown. This
   * operator may be used to specify an error_code object that should instead be
   * set with the asynchronous operation's result. For example:
   *
   * @code template <typename Handler>
   * void my_coroutine(basic_stackless_context<Handler> ctx)
   * {
   *   reenter (ctx)
   *   {
   *     ...
   *     yield my_socket.async_read_some(buffer, ctx[ec]);
   *     if (ec)
   *     {
   *       // An error occurred.
   *     }
   *     ...
   *   }
   * } @endcode
   */
  basic_stackless_context operator[](asio::error_code& ec) const
  {
    basic_stackless_context tmp(*this);
    tmp.ec_ = &ec;
    return tmp;
  }

#if defined(GENERATING_DOCUMENTATION)
private:
#endif // defined(GENERATING_DOCUMENTATION)
  detail::stackless_impl_base<Handler>** impl_;
  asio::error_code* ec_;
};

#if defined(GENERATING_DOCUMENTATION)
/// Context object that represents the currently executing coroutine.
typedef basic_stackless_context<unspecified> stackless_context;
#else // defined(GENERATING_DOCUMENTATION)
typedef basic_stackless_context<
  detail::wrapped_handler<
    io_service::strand, void(*)(),
    detail::is_continuation_if_running> > stackless_context;
#endif // defined(GENERATING_DOCUMENTATION)

/**
 * @defgroup go asio::go
 *
 * @brief Start a new stackless coroutine.
 *
 * The go() function is a high-level wrapper over the asio::coroutine
 * class. This function enables programs to implement asynchronous logic in a
 * synchronous manner, as illustrated by the following example:
 *
 * @code asio::go(my_strand, do_echo);
 *
 * // ...
 *
 * char data[128];
 * std::size_t length;
 *
 * // ...
 *
 * void do_echo(asio::stackless_context ctx)
 * {
 *   try
 *   {
 *     reenter (ctx)
 *     {
 *       for (;;)
 *       {
 *         await length = my_socket.async_read_some(
 *             asio::buffer(data), ctx);
 *
 *         await asio::async_write(my_socket,
 *             asio::buffer(data, length), ctx);
 *       }
 *     }
 *   }
 *   catch (std::exception& e)
 *   {
 *     // ...
 *   }
 * } @endcode
 */
/*@{*/

/// Start a new stackless coroutine, calling the specified handler when it
/// completes.
/**
 * This function is used to launch a new coroutine.
 *
 * @param handler A handler to be called when the coroutine exits. More
 * importantly, the handler provides an execution context (via the the handler
 * invocation hook) for the coroutine. The handler must have the signature:
 * @code void handler(); @endcode
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(basic_stackless_context<Handler> yield); @endcode
 */
template <typename Handler, typename Function>
void go(ASIO_MOVE_ARG(Handler) handler,
    ASIO_MOVE_ARG(Function) function);

/// Start a new stackless coroutine, inheriting the execution context of another.
/**
 * This function is used to launch a new coroutine.
 *
 * @param ctx Identifies the current coroutine as a parent of the new
 * coroutine. This specifies that the new coroutine should inherit the
 * execution context of the parent. For example, if the parent coroutine is
 * executing in a particular strand, then the new coroutine will execute in the
 * same strand.
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(basic_stackless_context<Handler> yield); @endcode
 */
template <typename Handler, typename Function>
void go(basic_stackless_context<Handler> ctx,
    ASIO_MOVE_ARG(Function) function);

/// Start a new stackless coroutine that executes in the context of a strand.
/**
 * This function is used to launch a new coroutine.
 *
 * @param strand Identifies a strand. By starting multiple coroutines on the
 * same strand, the implementation ensures that none of those coroutines can
 * execute simultaneously.
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(stackless_context yield); @endcode
 */
template <typename Function>
void go(asio::io_service::strand strand,
    ASIO_MOVE_ARG(Function) function);

/// Start a new stackless coroutine that executes on a given io_service.
/**
 * This function is used to launch a new coroutine.
 *
 * @param io_service Identifies the io_service that will run the coroutine. The
 * new coroutine is implicitly given its own strand within this io_service.
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(stackless_context yield); @endcode
 */
template <typename Function>
void go(asio::io_service& io_service,
    ASIO_MOVE_ARG(Function) function);

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/go.hpp"

#endif // ASIO_GO_HPP
