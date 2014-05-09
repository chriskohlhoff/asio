//
// chain.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_CHAIN_HPP
#define ASIO_CHAIN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Composes a chain of function objects into a continuation.
/**
 * The @c chain function is a variadic template function that composes one or
 * more functions into a single function object. This returned function object
 * is a continuation, which means that it has a @c void return type and may
 * only be called once.
 *
 * When we perform:
 *
 * @code asio::chain(t0, t1, ..., tN) @endcode
 *
 * the library turns the completion tokens @c t0 to @c tN into the
 * corresponding function objects @c f0 to @c fN by applying the @c
 * handler_type type trait. These function objects are then chained such that
 * they execute serially, and the return value of any given function is passed
 * as an argument to the next one.
 *
 * The returned continuation object automatically inherits the behavior of the
 * final completion token. For example, this means that the following works
 * as expected:
 *
 * @code auto c = asio::chain(
 *   []{ return 42; },
 *   [](int i){ return i * 2; },
 *   [](int i){ return std::to_string(i); },
 *   [](std::string s){ return "value is " + s; },
 *   asio::use_future);
 * std::future<int> fut = io_service.dispatch(c);
 * std::cout << fut.get() << std::endl; @endcode
 *
 * and outputs the string <tt>value is 84</tt>.
 */
#if defined(GENERATING_DOCUMENTATION)
template <typename... CompletionTokens>
unspecified chain(ASIO_MOVE_ARG(CompletionTokens)... tokens);
#endif // defined(GENERATING_DOCUMENTATION)

/// Composes a chain of function objects into a continuation.
/**
 * The @c chain function is a variadic template function that composes one or
 * more functions into a single function object. This returned function object
 * is a continuation, which means that it has a @c void return type and may
 * only be called once.
 *
 * This overload is used to specify the signature of the first function object
 * in the chain.
 *
 * When we perform:
 *
 * @code asio::chain<void(int)>(t0, t1, ..., tN) @endcode
 *
 * the library turns the completion tokens @c t0 to @c tN into the
 * corresponding function objects @c f0 to @c fN by applying the @c
 * handler_type type trait. These function objects are then chained such that
 * they execute serially, and the return value of any given function is passed
 * as an argument to the next one.
 */
#if defined(GENERATING_DOCUMENTATION)
template <typename Signature, typename... CompletionTokens>
unspecified chain(ASIO_MOVE_ARG(CompletionTokens)... tokens);
#endif // defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/chain.hpp"

#endif // ASIO_CHAIN_HPP
