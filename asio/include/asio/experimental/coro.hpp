//
// experimental/coro.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2021 Klemens D. Morgenstern
//                    (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_CORO_HPP
#define ASIO_EXPERIMENTAL_CORO_HPP

#include "asio/detail/config.hpp"
#include "asio/dispatch.hpp"
#include "asio/experimental/detail/coro_promise_allocator.hpp"
#include "asio/experimental/detail/partial_promise.hpp"
#include "asio/experimental/detail/coro_traits.hpp"
#include "asio/error.hpp"
#include "asio/error_code.hpp"
#include "asio/experimental/use_coro.hpp"
#include "asio/post.hpp"

#include "asio/detail/push_options.hpp"

namespace asio
{
namespace experimental
{

namespace detail
{

template <typename T, typename Coroutine >
struct coro_with_arg;

}


/// The main type of a resumable coroutine.
/**
 *
 * @tparam Yield The type or signature used by co_yield.
 * @tparam Return The type used for co_return.
 * @tparam Executor The underlying executor.
 */
template <typename Yield = void, typename Return = void,
    typename Executor = any_io_executor>
struct coro
{
  /// The traits of the coroutine. See asio::experimental::coro_traits for details.
  using traits = detail::coro_traits<Yield, Return, Executor>;

  /// The value that can be passed into a symmetrical cororoutine. Void if assymetrical.
  using input_type = typename traits::input_type;
  /// The type that can be passed out through a co_yield.
  using yield_type = typename traits::yield_type;
  /// The type that can be passed out through a co_return.
  using return_type = typename traits::return_type;
  /// The type received by a co_await or async_resume. It's a combination of yield and return.
  using result_type = typename traits::result_type;
  /// The signature used by the async_resume.
  using signature_type = typename traits::signature_type;
  /// Whether or not the coroutine is noexcept.
  constexpr static bool is_noexcept = traits::is_noexcept;
  /// The error type of the coroutine. Void for noexcept
  using error_type = typename traits::error_type;
  /// Completion handler type used by async_resume.
  using completion_handler = typename traits::completion_handler;

  /// The internal promise-type of the coroutine.
  using promise_type = detail::coro_promise<Yield, Return, Executor>;

#if !defined(GENERATING_DOCUMENTATION)
  template <typename T, typename Coroutine >
  friend struct detail::coro_with_arg;
#endif // !defined(GENERATING_DOCUMENTATION)
  /// The executor type.
  using executor_type = Executor;

#if !defined(GENERATING_DOCUMENTATION)
  friend struct detail::coro_promise<Yield, Return, Executor>;
#endif // !defined(GENERATING_DOCUMENTATION)

  /// The default constructor, gives an invalid coroutine.
  coro() = default;

  /// Move constructor.
  coro(coro&& lhs) noexcept
    : coro_(std::exchange(lhs.coro_, nullptr))
  {
  }

  coro(const coro &) = delete;

  /// Move assignment.
  coro& operator=(coro&& lhs) noexcept
  {
    std::swap(coro_, lhs.coro_);
    return *this;
  }

  coro& operator=(const coro&) = delete;

  /// Destructor. Destroys the coroutine, if it holds a valid one.
  /** \attention This does not cancel an active coroutine.
   * Destructing a resume couritine, i.e. one with a call to async_resume
   * that has not completed is undefined behaviour.
   */

  ~coro()
  {
    if (coro_ != nullptr)
    {
      struct destroyer
      {
        detail::coroutine_handle<promise_type> handle;

        destroyer(const detail::coroutine_handle<promise_type>& handle)
          : handle(handle)
        {
        }

        destroyer(destroyer&& lhs)
          : handle(std::exchange(lhs.handle, nullptr))
        {
        }

        destroyer(const destroyer&) = delete;

        void operator()() {}

        ~destroyer()
        {
          if (handle)
            handle.destroy();
        }
      };

      auto handle =
        detail::coroutine_handle<promise_type>::from_promise(*coro_);
      if (handle)
        asio::dispatch(coro_->get_executor(), destroyer{handle});
    }
  }
  /// Get the used executor.
  executor_type get_executor() const
  {
    if (coro_)
      return coro_->get_executor();

    if constexpr (std::is_default_constructible_v<Executor>)
      return Executor{};
    else
      throw std::logic_error("Coroutine has no executor");
  }
  /// Resume the coroutine.
  /**
   * \param token The completion token of the async resume.
   *
   * \attention Calling an invalid coroutine with a noexcept signature is undefined behaviour.
   *
   * \note This overload is only available for coroutines without an input value.
   */
  template <typename CompletionToken>
    requires std::is_void_v<input_type>
  auto async_resume(CompletionToken&& token)
  {
    return async_initiate<CompletionToken,
        typename traits::completion_handler>(
          initiate_async_resume(this), token);
  }

  /// Resume the coroutine.
  /**
   * \param token The completion token of the async resume.
   *
   * \attention Calling an invalid coroutine with a noexcept signature is undefined behaviour.
   *
   * \note This overload is only available for coroutines with an input value.
   */
  template <typename CompletionToken, detail::convertible_to<input_type> T>
  auto async_resume(T&& ip, CompletionToken&& token)
  {
    return async_initiate<CompletionToken,
        typename traits::completion_handler>(
          initiate_async_resume(this), token, std::forward<T>(ip));
  }

  /// Operator used for coroutines without input value.
  auto operator co_await() requires (std::is_void_v<input_type>)
  {
    return awaitable_t{*this};
  }

  /// Operator used for coroutines with  input value.
  /**
   *
   * @param ip The input value
   * @return An awaitable handle.
   *
   * \code
   * coro<void> push_values(coro<double(int)> c)
   * {
   *    std::optional<double> res = co_await c(42);
   * }
   * \endcode
   *
   */
  template <detail::convertible_to<input_type> T>
  auto operator()(T&& ip)
  {
    return detail::coro_with_arg<std::decay_t<T>, coro>{
        std::forward<T>(ip), *this};
  }
  /// Check whether the coroutine is open, i.e. can be resumed.
  bool is_open() const
  {
    if (coro_)
    {
      auto handle =
        detail::coroutine_handle<promise_type>::from_promise(*coro_);
      return handle && !handle.done();
    }
    else
      return false;
  }

  /// Check whether the coroutine is open, i.e. can be resumed.
  explicit operator bool() const { return is_open(); }

// TODO check if cancellation can be done
//  /// Send a cancel signal to the coroutine.
//  /**
//   * \note There is no guarantee that this actually cancels.
//   * The coroutine might ignore the signal.
//   *
//   * @param ct The canecllation type
//   */
//  void cancel(cancellation_type ct = cancellation_type::all)
//  {
//    if (is_open() && coro_->cancel && !coro_->cancel->state.cancelled())
//      asio::dispatch(get_executor(),
//          [ct, coro = coro_]
//          {
//            if (coro)
//            {
//
//              coro->cancel->state.force_emit(ct);
//            }
//          });
//  }

private:
  struct awaitable_t;

  struct initiate_async_resume;

  explicit coro(promise_type* const cr) : coro_(cr) {}

  promise_type* coro_{nullptr};
};

} // namespace experimental
} // namespace asio


#include "asio/experimental/impl/coro.hpp"

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_CORO_HPP
