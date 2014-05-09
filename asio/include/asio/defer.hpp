//
// defer.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DEFER_HPP
#define ASIO_DEFER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Submits a chain of function objects for execution using an executor's
/// @c defer() member function.
/**
 * The @c defer function is a variadic template function that submits a
 * chain of function objects for execution.
 *
 * When we perform:
 *
 * @code asio::defer(t0, t1, ..., tN) @endcode
 *
 * the library turns the completion tokens @c t0 to @c tN into the
 * corresponding function objects @c f0 to @c fN by applying the @c
 * handler_type type trait. The first function object is deferred to its
 * associated executor, and the remaining function objects are dispatched in
 * sequence. The return value of any given function is passed as an argument
 * to the next one. For example:
 *
 * @code std::future<std::string> fut = asio::defer(
 *   []{ return 42; },
 *   [](int i){ return i * 2; },
 *   [](int i){ return std::to_string(i); },
 *   [](std::string s){ return "value is " + s; },
 *   asio::use_future);
 * std::cout << fut.get() << std::endl; @endcode
 *
 * will output the string <tt>value is 84</tt>.
 */
#if defined(GENERATING_DOCUMENTATION)
template <typename... CompletionTokens>
void_or_deduced defer(ASIO_MOVE_ARG(CompletionTokens)... tokens);
#endif // defined(GENERATING_DOCUMENTATION)

/// Submits a chain of function objects for execution using an executor's
/// @c defer() member function.
/**
 * The @c defer function is a variadic template function that submits a
 * chain of function objects for execution.
 *
 * When we perform:
 *
 * @code asio::defer(ex, t0, t1, ..., tN) @endcode
 *
 * the library turns the completion tokens @c t0 to @c tN into the
 * corresponding function objects @c f0 to @c fN by applying the @c
 * handler_type type trait. The first function object is deferred on the
 * specified executor @c ex, and the remaining function objects are dispatched
 * in sequence. The return value of any given function is passed as an argument
 * to the next one. For example:
 *
 * @code system_executor ex;
 * std::future<std::string> fut = asio::defer(ex,
 *   []{ return 42; },
 *   [](int i){ return i * 2; },
 *   [](int i){ return std::to_string(i); },
 *   [](std::string s){ return "value is " + s; },
 *   asio::use_future);
 * std::cout << fut.get() << std::endl; @endcode
 *
 * will output the string <tt>value is 84</tt>.
 */
#if defined(GENERATING_DOCUMENTATION)
template <typename Executor, typename... CompletionTokens>
void_or_deduced defer(ASIO_MOVE_ARG(Executor) executor,
    ASIO_MOVE_ARG(CompletionTokens)... tokens);
#endif // defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/defer.hpp"

#endif // ASIO_DEFER_HPP
