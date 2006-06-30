//
// handler_dispatch_hook.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_HANDLER_DISPATCH_HOOK_HPP
#define ASIO_HANDLER_DISPATCH_HOOK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

namespace asio {

/// Default dispatch function for handlers.
/**
 * Completion handlers for asynchronous operations are invoked by the
 * io_service associated with the corresponding object (e.g. a socket or
 * deadline_timer). Certain guarantees are made on when the handler may be
 * invoked, in particular that a handler can only be invoked from a thread that
 * is currently calling asio::io_service::run() on the corresponding
 * asio::io_service object. Handlers may subsequently be dispatched through
 * other objects (such as asio::strand objects) that provide additional
 * guarantees.
 *
 * When asynchronous operations are composed from other asynchronous
 * operations, all intermediate handlers should be dispatched using the same
 * method as the final handler. This is required to ensure that user-defined
 * objects are not accessed in a way that may violate the guarantees. This
 * hooking function ensures that the dispatch method used for the final handler
 * is accessible at each intermediate step.
 *
 * Implement asio_handler_dispatch for your own handlers to specify a custom
 * dispatching strategy.
 *
 * This default implementation is simply:
 * @code
 * handler();
 * @endcode
 *
 * @par Example:
 * @code
 * class my_handler;
 *
 * template <typename Handler>
 * void asio_handler_dispatch(Handler handler, my_handler* context)
 * {
 *   context->strand_.dispatch(handler);
 * }
 * @endcode
 */
template <typename Handler>
inline void asio_handler_dispatch(Handler handler, ...)
{
  handler();
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_HANDLER_DISPATCH_HOOK_HPP
