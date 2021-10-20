//
// experimental/px.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
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
#include <asio/compose.hpp>
#include "asio/redirect_error.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"
#include "asio/experimental/coro.hpp"

#include <queue>
#include <tuple>

namespace asio
{
namespace experimental
{

namespace detail
{

template<typename T, typename U, typename Executor>
struct coro_spawn_initiate
{
  coro<T, U, Executor> c;

  template<typename ... Args>
  void operator()(auto & self)
  {
    auto cp = std::make_shared<coro<T, U, Executor>>(std::move(c));
    cp->async_resume(
            bind_executor(
              self.get_executor(),
              [cp, self = std::move(self)](auto ... res) mutable
              {
                self.complete(std::move(res)...);
              }));
  }
};


}

/// Spawn an asio::experimental::coro.
/**
 * This spawns the coroutine off for execution on it's executor. It binds the lifetime of the coroutine to the executor.
 *
 * @tparam T The return value of the coroutine
 * @tparam Executor The executor to be used
 * @tparam CompletionToken The type of the completion token, with the signature `void(std::exception_ptr, T)`.
 * @param c The coroutine
 * @param completion_token The completion token
 * @return Implementation defined
 */
template<typename T, typename Executor, typename CompletionToken>
auto co_spawn(coro<void, T, Executor> c, CompletionToken && completion_token)
{
  auto exec = c.get_executor();
  return asio::async_compose<CompletionToken, void(std::exception_ptr, T)>(
          detail::coro_spawn_initiate<void, T, Executor>{std::move(c)}, completion_token, exec);
}

/// Spawn an asio::experimental::coro.
/**
 * This spawns the coroutine off for execution on it's executor. It binds the lifetime of the coroutine to the executor.
 *
 * @tparam T The return value of the coroutine
 * @tparam Executor The executor to be used
 * @tparam CompletionToken The type of the completion token, with the signature `void(std::exception_ptr, T)`.
 * @param c The coroutine
 * @param completion_token The completion token
 * @return Implementation defined
 */

template<typename T, typename Executor, typename CompletionToken>
auto co_spawn(coro<void(), T, Executor> c, CompletionToken && completion_token)
{
  auto exec = c.get_executor();
  return asio::async_compose<CompletionToken, void(std::exception_ptr, T)>(
          detail::coro_spawn_initiate<void(), T, Executor>{std::move(c)}, completion_token, exec);
}

/// Spawn an asio::experimental::coro.
/**
 * This spawns the coroutine off for execution on it's executor. It binds the lifetime of the coroutine to the executor.
 *
 * @tparam T The return value of the coroutine
 * @tparam Executor The executor to be used
 * @tparam CompletionToken The type of the completion token, with the signature `void(T)`.
 * @param c The coroutine
 * @param completion_token The completion token
 * @return Implementation defined
 */
template<typename T, typename Executor, typename CompletionToken>
auto co_spawn(coro<void() noexcept, T, Executor> c, CompletionToken && completion_token)
{
  auto exec = c.get_executor();
  return asio::async_compose<CompletionToken, void(T)>(
          detail::coro_spawn_initiate<void() noexcept, T, Executor>{std::move(c)}, completion_token, exec);
}


/// Spawn an asio::experimental::coro.
/**
 * This spawns the coroutine off for execution on it's executor. It binds the lifetime of the coroutine to the executor.
 *
 * @tparam T The return value of the coroutine
 * @tparam Executor The executor to be used
 * @tparam CompletionToken The type of the completion token, with the signature `void(std::exception_ptr)`.
 * @param c The coroutine
 * @param completion_token The completion token
 * @return Implementation defined
 */
template<typename Executor, typename CompletionToken>
auto co_spawn(coro<void, void, Executor> c, CompletionToken && completion_token)
{
  auto exec = c.get_executor();
  return asio::async_compose<CompletionToken, void(std::exception_ptr)>(
          detail::coro_spawn_initiate<void, void, Executor>{std::move(c)}, completion_token, exec);
}

/// Spawn an asio::experimental::coro.
/**
 * This spawns the coroutine off for execution on it's executor. It binds the lifetime of the coroutine to the executor.
 *
 * @tparam T The return value of the coroutine
 * @tparam Executor The executor to be used
 * @tparam CompletionToken The type of the completion token, with the signature `void(std::exception_ptr)`.
 * @param c The coroutine
 * @param completion_token The completion token
 * @return Implementation defined
 */
template<typename Executor, typename CompletionToken>
auto co_spawn(coro<void(), void, Executor> c, CompletionToken && completion_token)
{
  auto exec = c.get_executor();
  return asio::async_compose<CompletionToken, void(std::exception_ptr)>(
          detail::coro_spawn_initiate<void(), void, Executor>{std::move(c)}, completion_token, exec);
}

/// Spawn an asio::experimental::coro.
/**
 * This spawns the coroutine off for execution on it's executor. It binds the lifetime of the coroutine to the executor.
 *
 * @tparam T The return value of the coroutine
 * @tparam Executor The executor to be used
 * @tparam CompletionToken The type of the completion token, with the signature `void()`.
 * @param c The coroutine
 * @param completion_token The completion token
 * @return Implementation defined
 */
template<typename Executor, typename CompletionToken>
auto co_spawn(coro<void() noexcept, void, Executor> c, CompletionToken && completion_token)
{
  auto exec = c.get_executor();
  return asio::async_compose<CompletionToken, void()>(
          detail::coro_spawn_initiate<void() noexcept, void, Executor>{std::move(c)}, completion_token, exec);
}

}
}

#include "asio/detail/pop_options.hpp"

#endif //ASIO_EXPERIMENTAL_CO_SPAWN_HPP
