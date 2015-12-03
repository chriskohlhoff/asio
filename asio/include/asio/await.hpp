//
// await.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_AWAIT_HPP
#define ASIO_AWAIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_CO_AWAIT)

#include <experimental/resumable>
#include "asio/executor.hpp"
#include "asio/strand.hpp"

#ifndef co_await
# define co_await await
#endif

#ifndef co_return
# define co_return if (0) { \
    co_await std::experimental::suspend_never(); \
  } else return
#endif

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class awaiter;
class awaitee_base;
template <typename> class awaitee;
template <typename, typename> class await_handler_base;
template <typename> class make_await_context;
using std::experimental::coroutine_handle;

} // namespace detail

/// Context object that represents the currently executing coroutine.
/**
 * The basic_unsynchronized_await_context class is used to represent the
 * currently executing coroutine. A basic_unsynchronized_await_context may be
 * passed as a handler to an asynchronous operation. For example:
 *
 * @code template <typename Executor>
 * void my_coroutine(basic_unsynchronized_await_context<Executor> ctx)
 * {
 *   ...
 *   std::size_t n = co_await my_socket.async_read_some(buffer, ctx);
 *   ...
 * } @endcode
 *
 * The initiating function (async_read_some in the above example) suspends the
 * current coroutine. The coroutine is resumed when the asynchronous operation
 * completes, and the result of the operation is returned.
 */
template <typename Executor>
class basic_unsynchronized_await_context
{
public:
  /// Copy constructor.
  basic_unsynchronized_await_context(
      const basic_unsynchronized_await_context&) = default;

  /// Move constructor.
  basic_unsynchronized_await_context(
      basic_unsynchronized_await_context&&) = default;

  // No assignment allowed.
  basic_unsynchronized_await_context& operator=(
      const basic_unsynchronized_await_context&) = delete;

  /// Construct from another context type.
  template <typename E>
  basic_unsynchronized_await_context(
      const basic_unsynchronized_await_context<E>& other) noexcept
    : ex_(other.ex_),
      awaiter_(other.awaiter_)
  {
  }

  /// Move constructor from another context type.
  template <typename E>
  basic_unsynchronized_await_context(
      basic_unsynchronized_await_context<E>&& other) noexcept
    : ex_(std::move(other.ex_)),
      awaiter_(std::exchange(other.awaiter_, nullptr))
  {
  }

  /// The associated executor type.
  typedef Executor executor_type;

  /// Get the associated executor.
  executor_type get_executor() const noexcept { return ex_; }

private:
  template <typename> friend class basic_unsynchronized_await_context;
  template <typename, typename> friend class detail::await_handler_base;
  friend class detail::make_await_context<Executor>;

  // Private constructor used by make_await_context.
  basic_unsynchronized_await_context(Executor ex, detail::awaiter* a)
    : ex_(ex),
      awaiter_(a)
  {
  }

  Executor ex_;
  detail::awaiter* awaiter_;
};

/// Alias for a context that automatically uses a strand.
template <typename Executor>
using basic_await_context =
  basic_unsynchronized_await_context<strand<Executor>>;

/// Typedef for most simplest use, using a type-erased executor.
typedef basic_await_context<executor> await_context;

/// The return type of a coroutine or asynchronous operation.
template <typename T>
class awaitable
{
public:
  // Construct the awaitable from a coroutine's promise object.
  explicit awaitable(detail::awaitee<T>* a) : awaitee_(a) {}

  /// Move constructor.
  awaitable(awaitable&& other)
    : awaitee_(std::exchange(other.awaitee_, nullptr))
  {
  }

  // Not copy constructible or copy assignable.
  awaitable(const awaitable&) = delete;
  awaitable& operator=(const awaitable&) = delete;

  /// Destructor
  ~awaitable();

  // Support for co_await keyword.
  bool await_ready();
  void await_suspend(detail::coroutine_handle<detail::awaiter>);
  template <class U>
  void await_suspend(detail::coroutine_handle<detail::awaitee<U>>);
  T await_resume();

private:
  template <class, class> friend class detail::await_handler_base;
  detail::awaitee<T>* awaitee_;
};

/// Spawn a new thread of execution.
template <typename F, typename E, typename... Args>
void spawn(F f, const basic_unsynchronized_await_context<E>& ctx,
    Args&&... args);

/// Spawn a new thread of execution.
template <typename F, typename Executor, typename... Args>
auto spawn(F f, Executor ex, Args&&... args)
  -> typename enable_if<is_executor<Executor>::value>::type;

/// Spawn a new thread of execution.
template <typename F, typename ExecutionContext, typename... Args>
auto spawn(F f, ExecutionContext& ctx, Args&&... args)
  -> typename enable_if<is_convertible<
      ExecutionContext&, execution_context&>::value>::type;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/await.hpp"

#endif // defined(ASIO_HAS_CO_AWAIT)

#endif // ASIO_AWAIT_HPP
