//
// experimental/co_spawn.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern
//                    (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef ASIO_EXPERIMENTAL_CO_SPAWN_HPP
#define ASIO_EXPERIMENTAL_CO_SPAWN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <utility>
#include "asio/compose.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/experimental/coro.hpp"
#include "asio/experimental/prepend.hpp"
#include "asio/redirect_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {

/// Spawn a resumable coroutine.
/**
 * This function spawns the coroutine for execution on its executor. It binds
 * the lifetime of the coroutine to the executor.
 *
 * @param c The coroutine
 *
 * @param token The completion token
 *
 * @returns Implementation defined
 */
template <typename T, typename Executor, typename CompletionToken>
ASIO_INITFN_AUTO_RESULT_TYPE(
    CompletionToken, void(std::exception_ptr, T))
co_spawn(coro<void, T, Executor> c, CompletionToken&& token)
{
  std::move(c).async_resume(std::forward<CompletionToken>(token));
}

/// Spawn a resumable coroutine.
/**
 * This function spawns the coroutine for execution on its executor. It binds
 * the lifetime of the coroutine to the executor.
 *
 * @param c The coroutine
 *
 * @param token The completion token
 *
 * @returns Implementation defined
 */
template <typename T, typename Executor, typename CompletionToken>
ASIO_INITFN_AUTO_RESULT_TYPE(
    CompletionToken, void(std::exception_ptr, T))
co_spawn(coro<void(), T, Executor> c, CompletionToken&& token)
{
  std::move(c).async_resume(std::forward<CompletionToken>(token));
}

/// Spawn a resumable coroutine.
/**
 * This function spawns the coroutine for execution on its executor. It binds
 * the lifetime of the coroutine to the executor.
 *
 * @param c The coroutine
 *
 * @param token The completion token
 *
 * @returns Implementation defined
 */
template <typename T, typename Executor, typename CompletionToken>
ASIO_INITFN_AUTO_RESULT_TYPE(CompletionToken, void(T))
co_spawn(coro<void() noexcept, T, Executor> c, CompletionToken&& token)
{
  std::move(c).async_resume(std::forward<CompletionToken>(token));
}

/// Spawn a resumable coroutine.
/**
 * This function spawns the coroutine for execution on its executor. It binds
 * the lifetime of the coroutine to the executor.
 *
 * @param c The coroutine
 *
 * @param token The completion token
 *
 * @returns Implementation defined
 */
template <typename Executor, typename CompletionToken>
ASIO_INITFN_AUTO_RESULT_TYPE(
    CompletionToken, void(std::exception_ptr))
co_spawn(coro<void, void, Executor> c, CompletionToken&& token)
{
  std::move(c).async_resume(std::forward<CompletionToken>(token));
}

/// Spawn a resumable coroutine.
/**
 * This function spawns the coroutine for execution on its executor. It binds
 * the lifetime of the coroutine to the executor.
 *
 * @param c The coroutine
 *
 * @param token The completion token
 *
 * @returns Implementation defined
 */
template <typename Executor, typename CompletionToken>
ASIO_INITFN_AUTO_RESULT_TYPE(
    CompletionToken, void(std::exception_ptr))
co_spawn(coro<void(), void, Executor> c, CompletionToken&& token)
{
  std::move(c).async_resume(std::forward<CompletionToken>(token));
}

/// Spawn a resumable coroutine.
/**
 * This function spawns the coroutine for execution on its executor. It binds
 * the lifetime of the coroutine to the executor.
 *
 * @param c The coroutine
 *
 * @param token The completion token
 *
 * @returns Implementation defined
 */
template <typename Executor, typename CompletionToken>
ASIO_INITFN_AUTO_RESULT_TYPE(CompletionToken, void())
co_spawn(coro<void() noexcept, void, Executor> c, CompletionToken&& token)
{
  std::move(c).async_resume(std::forward<CompletionToken>(token));
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif //ASIO_EXPERIMENTAL_CO_SPAWN_HPP
