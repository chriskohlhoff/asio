//
// experimental/use_coro.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern
//                    (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_USE_CORO_HPP
#define ASIO_EXPERIMENTAL_USE_CORO_HPP

#include "asio/detail/config.hpp"
#include <optional>
#include "asio/bind_cancellation_slot.hpp"
#include "asio/bind_executor.hpp"
#include "asio/error_code.hpp"
#include "asio/experimental/detail/partial_promise.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

class any_io_executor;

namespace experimental {

/// A completion token that represents the currently executing coroutine.
/**
 * The @c use_coro_t class, with its value @c use_coro, is used to
 * represent an operation that can be awaited by the current coro..
 * This completion token may be passed as a handler to an asynchronous operation.
 * For example:
 *
 * @code coro<void> my_coroutine(tcp::socket my_socket)
 * {
 *   std::size_t n = co_await my_socket.async_read_some(buffer, use_coro);
 *   ...
 * } @endcode
 *
 * When used with co_await, the initiating function (@c async_read_some in the
 * above example) suspends the current coroutine. The coroutine is resumed when
 * the asynchronous operation completes, and the result of the operation is
 * returned.
 */
template <typename Executor = any_io_executor>
struct use_coro_t
{
  /// Default constructor.
  ASIO_CONSTEXPR use_coro_t(
#if defined(ASIO_ENABLE_HANDLER_TRACKING)
# if defined(ASIO_HAS_SOURCE_LOCATION)
      detail::source_location location = detail::source_location::current()
# endif // defined(ASIO_HAS_SOURCE_LOCATION)
#endif // defined(ASIO_ENABLE_HANDLER_TRACKING)
  )
#if defined(ASIO_ENABLE_HANDLER_TRACKING)
  # if defined(ASIO_HAS_SOURCE_LOCATION)
    : file_name_(location.file_name()),
      line_(location.line()),
      function_name_(location.function_name())
# else // defined(ASIO_HAS_SOURCE_LOCATION)
    : file_name_(0),
      line_(0),
      function_name_(0)
# endif // defined(ASIO_HAS_SOURCE_LOCATION)
#endif // defined(ASIO_ENABLE_HANDLER_TRACKING)
  {
  }

  /// Constructor used to specify file name, line, and function name.
  ASIO_CONSTEXPR use_coro_t(const char* file_name,
                            int line, const char* function_name)
#if defined(ASIO_ENABLE_HANDLER_TRACKING)
  : file_name_(file_name),
      line_(line),
      function_name_(function_name)
#endif // defined(ASIO_ENABLE_HANDLER_TRACKING)
  {
#if !defined(ASIO_ENABLE_HANDLER_TRACKING)
    (void)file_name;
    (void)line;
    (void)function_name;
#endif // !defined(ASIO_ENABLE_HANDLER_TRACKING)
  }

  /// Adapts an executor to add the @c use_coro_t completion token as the
  /// default.
  template <typename InnerExecutor>
  struct executor_with_default : InnerExecutor
  {
    /// Specify @c use_coro_t as the default completion token type.
    typedef use_coro_t default_completion_token_type;

    /// Construct the adapted executor from the inner executor type.
    template <typename InnerExecutor1>
    executor_with_default(const InnerExecutor1& ex,
                          typename constraint<
                                  conditional<
                                          !is_same<InnerExecutor1, executor_with_default>::value,
                                          is_convertible<InnerExecutor1, InnerExecutor>,
                                          false_type
                                  >::type::value
                          >::type = 0) ASIO_NOEXCEPT
            : InnerExecutor(ex)
    {
    }
  };

    /// Type alias to adapt an I/O object to use @c use_coro_t as its
    /// default completion token type.
#if defined(ASIO_HAS_ALIAS_TEMPLATES) \
  || defined(GENERATING_DOCUMENTATION)
    template <typename T>
  using as_default_on_t = typename T::template rebind_executor<
      executor_with_default<typename T::executor_type> >::other;
#endif // defined(ASIO_HAS_ALIAS_TEMPLATES)
    //   || defined(GENERATING_DOCUMENTATION)

    /// Function helper to adapt an I/O object to use @c use_coro_t as its
    /// default completion token type.
    template <typename T>
    static typename decay<T>::type::template rebind_executor<
            executor_with_default<typename decay<T>::type::executor_type>
    >::other
    as_default_on(ASIO_MOVE_ARG(T) object)
    {
      return typename decay<T>::type::template rebind_executor<
              executor_with_default<typename decay<T>::type::executor_type>
      >::other(ASIO_MOVE_CAST(T)(object));
    }
#if defined(ASIO_ENABLE_HANDLER_TRACKING)
  const char* file_name_;
  int line_;
  const char* function_name_;
#endif // defined(ASIO_ENABLE_HANDLER_TRACKING)
};


/// A completion token object that represents the currently executing coroutine.
/**
 * See the documentation for asio::use_coro_t for a usage example.
 */
#if defined(GENERATING_DOCUMENTATION)
constexpr use_coro_t<> use_coro;
#elif defined(ASIO_HAS_CONSTEXPR)
constexpr use_coro_t<> use_coro(0, 0, 0);
#elif defined(ASIO_MSVC)
__declspec(selectany) use_coro_t<> use_coro(0, 0, 0);
#endif


template <typename Yield, typename Return, typename Executor>
struct coro;

namespace detail {

template <typename Yield, typename Return, typename Executor>
struct coro_promise;

template <typename Executor, typename... Ts>
struct coro_init_handler
{
  struct handler_t
  {
  };

  constexpr static handler_t handler{};

  struct init_helper;

  struct promise_type
  {
    auto initial_suspend() noexcept { return suspend_always{}; }

    auto final_suspend() noexcept { return suspend_always(); }

    void return_void() {}

    void unhandled_exception() { assert(false); }

    auto await_transform(handler_t)
    {
      assert(executor);
      assert(h);
      return init_helper{this};
    }

    std::optional<Executor> executor;
    std::optional<std::tuple<Ts...>> result;
    coroutine_handle<> h;

    coro_init_handler get_return_object() { return coro_init_handler{this}; }

    cancellation_slot cancel_slot;
  };

  struct init_helper
  {
    promise_type *self_;

    constexpr static bool await_ready() noexcept { return true; }

    constexpr static void await_suspend(coroutine_handle<>) noexcept {}

    auto await_resume() const noexcept
    {
      assert(self_);
      return bind_cancellation_slot(self_->cancel_slot,
          bind_executor(*self_->executor, [self = self_](Ts... ts)
          {
            self->result.emplace(std::move(ts)...);
            self->h.resume();
          }));
    }
  };

  promise_type* promise;

  void unhandled_exception() noexcept
  {
    throw;
  }

  struct noexcept_version
  {
    promise_type *promise;

    constexpr static bool await_ready() noexcept { return false; }

    template <typename Yield, typename Return,
        convertible_to<Executor> Executor1>
    auto await_suspend(
        coroutine_handle<coro_promise<Yield, Return, Executor1> > h) noexcept
    {
      promise->executor = h.promise().get_executor();
      promise->h = h;
      return coroutine_handle<promise_type>::from_promise(*promise);
    }

    template <typename... Args>
    static auto resume_impl(std::tuple<Args...>&& tup)
    {
      return std::move(tup);
    }

    template <typename Arg>
    static auto resume_impl(std::tuple<Arg>&& tup)
    {
      return get<0>(std::move(tup));
    }

    static void resume_impl(std::tuple<>&&) {}

    auto await_resume() const noexcept
    {
      auto res = std::move(promise->result.value());
      coroutine_handle<promise_type>::from_promise(*promise).destroy();
      return resume_impl(std::move(res));
    }
  };

  struct throwing_version
  {
    promise_type *promise;

    constexpr static bool await_ready() noexcept { return false; }

    template <typename Yield, typename Return,
        convertible_to<Executor> Executor1>
    auto await_suspend(
        coroutine_handle<coro_promise<Yield, Return, Executor1> > h) noexcept
    {
      promise->executor = h.promise().get_executor();
      promise->h = h;
      return coroutine_handle<promise_type>::from_promise(*promise);
    }

    template <typename... Args>
    static auto resume_impl(std::tuple<Args...>&& tup)
    {
      return std::move(tup);
    }

    static void resume_impl(std::tuple<>&&) {}

    template <typename Arg>
    static auto resume_impl(std::tuple<Arg>&& tup)
    {
      return get<0>(std::move(tup));
    }

    template <typename... Args>
    static auto resume_impl(std::tuple<std::exception_ptr, Args...>&& tup)
    {
      auto ex = get<0>(std::move(tup));
      if (ex)
        std::rethrow_exception(ex);

      if constexpr (sizeof...(Args) == 0u)
        return;
      else if constexpr (sizeof...(Args) == 1u)
        return get<1>(std::move(tup));
      else
      {
        return
          [&]<std::size_t... Idx>(std::index_sequence<Idx...>)
          {
            return std::make_tuple(std::get<Idx + 1>(std::move(tup))...);
          }(std::make_index_sequence<sizeof...(Args) - 1>{});
      }
    }

    template <typename... Args>
    static auto resume_impl(
        std::tuple<asio::error_code, Args...>&& tup)
    {
      auto ec = get<0>(std::move(tup));
      if (ec)
        asio::detail::throw_exception(
            asio::system_error(ec, "error_code in use_coro"));

      if constexpr (sizeof...(Args) == 0u)
        return;
      else if constexpr (sizeof...(Args) == 1u)
        return get<1>(std::move(tup));
      else
        return
          [&]<std::size_t... Idx>(std::index_sequence<Idx...>)
          {
            return std::make_tuple(std::get<Idx + 1>(std::move(tup))...);
          }(std::make_index_sequence<sizeof...(Args) - 1>{});
    }

    static auto resume_impl(std::tuple<std::exception_ptr>&& tup)
    {
      auto ex = get<0>(std::move(tup));
      if (ex)
        std::rethrow_exception(ex);
    }

    static auto resume_impl(
        std::tuple<asio::error_code>&& tup)
    {
      auto ec = get<0>(std::move(tup));
      if (ec)
        asio::detail::throw_error(ec, "error_code in use_coro");
    }

    auto await_resume() const
    {
      auto res = std::move(promise->result.value());
      coroutine_handle<promise_type>::from_promise(*promise).destroy();
      return resume_impl(std::move(res));
    }
  };

  auto as_noexcept(cancellation_slot&& sl) && noexcept
  {
    promise->cancel_slot = std::move(sl);
    return noexcept_version{promise};
  }

  auto as_throwing(cancellation_slot&& sl) && noexcept
  {
    promise->cancel_slot = std::move(sl);
    return throwing_version{promise};
  }
};

} // namespace detail
} // namespace experimental

#if !defined(GENERATING_DOCUMENTATION)

template <typename Executor, typename R, typename... Args>
struct async_result<experimental::use_coro_t<Executor>, R(Args...)>
{
  using return_type = experimental::detail::coro_init_handler<
    Executor, typename decay<Args>::type...>;

  template <typename Initiation, typename... InitArgs>
  static return_type initiate(Initiation initiation,
      experimental::use_coro_t<Executor>, InitArgs... args)
  {
    std::move(initiation)(co_await return_type::handler, std::move(args)...);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/experimental/coro.hpp"

#endif // ASIO_EXPERIMENTAL_USE_CORO_HPP
