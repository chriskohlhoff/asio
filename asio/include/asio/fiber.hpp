// Copyright (c) 2021 Klemens D. Morgenstern
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ASIO_FIBER_HPP
#define ASIO_FIBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <boost/context/fiber.hpp>
#include <optional>

#include "asio/any_io_executor.hpp"
#include "asio/associated_cancellation_slot.hpp"
#include "asio/associated_executor.hpp"
#include "asio/bind_cancellation_slot.hpp"
#include "asio/bind_executor.hpp"
#include "asio/cancellation_state.hpp"
#include "asio/experimental/deferred.hpp"
#include "asio/dispatch.hpp"
#include "asio/is_executor.hpp"
#include "asio/post.hpp"

#include "asio/detail/push_options.hpp"

namespace asio
{

namespace detail
{

struct fiber_handle
{
  bool did_suspend{false};
  boost::context::fiber fiber;
  std::unique_ptr<fiber_handle> lifetime;
  cancellation_slot  cancel_slot;
  cancellation_state cancel_state{cancel_slot};

  fiber_handle(cancellation_slot  cancel_slot) : cancel_slot(std::move(cancel_slot)) {}
};

}
/** The basic context handle of a fiber.
 *
 * @tparam Executor The executor of the fiber_context.
*/
template<typename Executor = asio::any_io_executor>
struct basic_fiber_context
{
  /// The executor of the fiber.
  using executor_type = Executor;

  /// Copy construct a basic_fiber_context from anoter context. Can convert the executor type.
  template<typename Executor2>
  basic_fiber_context(
          const basic_fiber_context<Executor2>  & other,
          typename constraint<std::is_convertible_v<Executor2, Executor>>::type = 0) noexcept
          : handle_(other.handle_), fiber_(other.fiber_), executor_(other.get_executor())
  {
  }

  //// move construct a basic_fiber_context from anoter context. Can convert the executor type.
  template<typename Executor2>
  basic_fiber_context(
          basic_fiber_context<Executor2>  && other,
          typename constraint<std::is_convertible_v<Executor2, Executor>>::type = 0) noexcept
          : handle_(other.handle_), fiber_(std::move(other.fiber_)), executor_(std::move(other.get_executor()))
  {
  }
#if !defined(GENERATING_DOCUMENTATION)

  basic_fiber_context(detail::fiber_handle &handle,
                      boost::context::fiber && fiber,
                      executor_type exec) : handle_(&handle), fiber_(&fiber), executor_(std::move(exec))
  {
  }

#endif

  /// Get the executor of the fiber.
  executor_type get_executor() const {return executor_;}

  /// Get the cancellation state of the fiber.
  cancellation_state get_cancellation_state() const ASIO_NOEXCEPT
  {
    return handle_->cancel_state;
  }

  /// Reset the cancellation state of the fiber.
  void reset_cancellation_state()
  {
    handle_->cancel_state = cancellation_state(handle_->cancel_slot);
  }

  /// Reset the cancellation state of the fiber and insert a filter.
  template <typename Filter>
  void reset_cancellation_state(ASIO_MOVE_ARG(Filter) filter)
  {
    handle_->cancel_state = cancellation_state(handle_->cancel_slot, ASIO_MOVE_CAST(Filter)(filter));
  }

  /// Reset the cancellation state of the fiber and insert in and out filters.
  template <typename InFilter, typename OutFilter>
  void reset_cancellation_state(ASIO_MOVE_ARG(InFilter) in_filter,
                                ASIO_MOVE_ARG(OutFilter) out_filter)
  {
    handle_->cancel_state = cancellation_state(handle_->cancel_slot,
                                              ASIO_MOVE_CAST(InFilter)(in_filter),
                                              ASIO_MOVE_CAST(OutFilter)(out_filter));
  }

 public:

  template<typename H, ASIO_COMPLETION_SIGNATURES_TPARAMS>
  friend class async_result;

  detail::fiber_handle *handle_;
  // innter fiber
  boost::context::fiber *fiber_;
  executor_type executor_;

};

/// The fiber context type with the default executor.
typedef basic_fiber_context<> fiber_context;

namespace detail
{

template<typename Executor>
struct initiate_fiber
{

  template<typename CompletionHandler, typename Func>
  void operator()(CompletionHandler && completion_handler, Func && function, void(*)()) const
  {
    auto alloc = get_associated_executor(completion_handler, get_executor());

    struct fiber_runner : fiber_handle
    {
      Executor executor;
      Func func;
      CompletionHandler handler;

      fiber_runner(cancellation_slot slot, Executor executor, Func && func, CompletionHandler && handler)
          : fiber_handle(slot), executor(std::move(executor)),
            func(std::forward<Func>(func)), handler(std::forward<CompletionHandler>(handler)) {}

      auto complete()
      {
        auto def = asio::experimental::deferred(
                [this, lf = std::exchange(lifetime, nullptr)] () mutable
                {
                  return asio::experimental::deferred.template values();
                });
        if (this->did_suspend)
          asio::dispatch(get_associated_executor(handler, this->executor), std::move(def))(std::move(handler));
        else
          asio::post(get_associated_executor(handler, this->executor), std::move(def))(std::move(handler));

      }

      auto work(boost::context::fiber && fb) -> boost::context::fiber
      {
        using fiber_context = basic_fiber_context<Executor>;
        assert(lifetime);
        func(fiber_context{*this, std::move(fb), executor});
        complete();

        return std::move(fb);
      }

      void start(std::unique_ptr<fiber_runner> p)
      {
        fiber = boost::context::fiber([this](boost::context::fiber && fb) {return work(std::move(fb));});
        asio::dispatch(
                this->executor,
                [this, p = std::move(p)]() mutable
                {
                  lifetime = std::move(p);
                  fiber = std::move(fiber).resume();
                });

      }
    };

    auto runner = std::make_unique<fiber_runner>(get_associated_cancellation_slot(completion_handler),
                                                 get_executor(),
                                                 std::forward<Func>(function),
                                                 std::forward<CompletionHandler>(completion_handler));
    runner->start(std::move(runner));
  }

  template<typename CompletionHandler, typename Func, typename T>
  void operator()(CompletionHandler && completion_handler, Func && function, void(*)(T)) const
  {
    auto alloc = get_associated_executor(completion_handler, get_executor());

    struct fiber_runner : fiber_handle
    {
      Executor executor;
      Func func;
      CompletionHandler handler;

      fiber_runner(cancellation_slot slot, Executor executor, Func && func, CompletionHandler && handler)
          : fiber_handle(slot), executor(std::move(executor)),
            func(std::forward<Func>(func)), handler(std::forward<CompletionHandler>(handler)) {}

      auto complete(T && val = {})
      {
        auto def = asio::experimental::deferred(
                [this, lf = std::exchange(lifetime, nullptr), val = std::move(val)] () mutable
                {
                  return asio::experimental::deferred.template values(std::move(val));
                });

        if (this->did_suspend)
          asio::dispatch(get_associated_executor(handler, this->executor), std::move(def))(std::move(handler));
        else
          asio::post(get_associated_executor(handler, this->executor), std::move(def))(std::move(handler));

      }

      auto work(boost::context::fiber && fb) -> boost::context::fiber
      {
        using fiber_context = basic_fiber_context<Executor>;

        assert(lifetime);
        complete(func(fiber_context{*this, std::move(fb), executor}));
        return std::move(fb);
      }

      void start(std::unique_ptr<fiber_runner> p)
      {
        fiber = boost::context::fiber([this](boost::context::fiber && fb) {return work(std::move(fb));});
        asio::dispatch(
                this->executor,
                [this, p = std::move(p)]() mutable
                {
                  lifetime = std::move(p);
                  fiber = std::move(fiber).resume();
                });

      }
    };

    auto runner = std::make_unique<fiber_runner>(get_associated_cancellation_slot(completion_handler),
                                                 get_executor(),
                                                 std::forward<Func>(function),
                                                 std::forward<CompletionHandler>(completion_handler));
    runner->start(std::move(runner));
  }

  template<typename CompletionHandler, typename Func>
  void operator()(CompletionHandler && completion_handler, Func && function, void(*)(std::exception_ptr)) const
  {
    auto alloc = get_associated_executor(completion_handler, get_executor());

    struct fiber_runner : fiber_handle
    {
      Executor executor;
      Func func;
      CompletionHandler handler;

      fiber_runner(cancellation_slot slot, Executor executor, Func && func, CompletionHandler && handler)
              : fiber_handle(slot), executor(std::move(executor)),
                func(std::forward<Func>(func)), handler(std::forward<CompletionHandler>(handler)) {}

      auto complete(std::exception_ptr e)
      {
        auto def = asio::experimental::deferred(
                [this, lf = std::exchange(lifetime, nullptr), e = std::move(e)] () mutable
                {
                  return asio::experimental::deferred.template values(std::move(e));
                });

        if (this->did_suspend)
          asio::dispatch(get_associated_executor(handler, this->executor), std::move(def))(std::move(handler));
        else
          asio::post(get_associated_executor(handler, this->executor), std::move(def))(std::move(handler));

      }

      auto work(boost::context::fiber && fb) -> boost::context::fiber
      {
        using fiber_context = basic_fiber_context<Executor>;
        std::exception_ptr e;
        try {
          assert(lifetime);
          func(fiber_context{*this, std::move(fb), executor});
          complete(std::exception_ptr{});
        }
        catch (...)
        {
          e = std::current_exception();
        }
        if (e)
          complete(e);

        return std::move(fb);

      }

      void start(std::unique_ptr<fiber_runner> p)
      {
        fiber = boost::context::fiber([this](boost::context::fiber && fb) {return work(std::move(fb));});
        asio::dispatch(
                this->executor,
                [this, p = std::move(p)]() mutable
                {
                  lifetime = std::move(p);
                  fiber = std::move(fiber).resume();
                });

      }
    };

    auto runner = std::make_unique<fiber_runner>(get_associated_cancellation_slot(completion_handler),
                                                 get_executor(),
                                                 std::forward<Func>(function),
                                                 std::forward<CompletionHandler>(completion_handler));
    runner->start(std::move(runner));
  }

  template<typename CompletionHandler, typename Func, typename T>
  void operator()(CompletionHandler && completion_handler, Func && function, void(*)(std::exception_ptr, T)) const
  {
    auto alloc = get_associated_executor(completion_handler, get_executor());

    struct fiber_runner : fiber_handle
    {
      Executor executor;
      Func func;
      CompletionHandler handler;

      fiber_runner(cancellation_slot slot, Executor executor, Func && func, CompletionHandler && handler)
              : fiber_handle(slot), executor(std::move(executor)),
                func(std::forward<Func>(func)), handler(std::forward<CompletionHandler>(handler)) {}

      auto complete(std::exception_ptr e, T && val = {})
      {

        auto def = asio::experimental::deferred(
                [this, lf = std::exchange(lifetime, nullptr), e = std::move(e), val = std::move(val)] () mutable
                {
                  return asio::experimental::deferred.template values(std::move(e), std::move(val));
                });

        if (this->did_suspend)
          asio::dispatch(get_associated_executor(handler, this->executor), std::move(def))(std::move(handler));
        else
          asio::post(get_associated_executor(handler, this->executor), std::move(def))(std::move(handler));
      }

      auto work(boost::context::fiber && fb) -> boost::context::fiber
      {
        using fiber_context = basic_fiber_context<Executor>;
        std::exception_ptr e;

        try {
          assert(lifetime);
          complete(std::exception_ptr{}, func(fiber_context{*this, std::move(fb), executor}));
        }
        catch (...)
        {
          e = std::current_exception();
        }
        if (e)
          complete(e);

        return std::move(fb);
      }

      void start(std::unique_ptr<fiber_runner> p)
      {
        fiber = boost::context::fiber([this](boost::context::fiber && fb) {return work(std::move(fb));});
        asio::dispatch(
                this->executor,
                [this, p = std::move(p)]() mutable
                {
                  lifetime = std::move(p);
                  fiber = std::move(fiber).resume();
                });

      }
    };

    auto runner = std::make_unique<fiber_runner>(get_associated_cancellation_slot(completion_handler),
                                                 get_executor(),
                                                 std::forward<Func>(function),
                                                 std::forward<CompletionHandler>(completion_handler));
    runner->start(std::move(runner));
  }

  using executor_type = Executor;
    executor_type get_executor() const noexcept {return executor_;}

    initiate_fiber(executor_type exec) : executor_(exec) {}
 private:
    executor_type executor_;
};

}

/// Run a fiber based thread of execution.
/** This is an overload that can throw exception and return a value. The type is deduced from the function signature.
 *
 * @param ctx The execution context
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr, void)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket)
 * {
 *   std::size_t bytes_transferred = 0;
 *
 *   char data[1024];
 *   for (;;)
 *   {
 *     std::size_t n = socket.async_read_some(
 *         asio::buffer(data), ctx);
 *
 *     asio::async_write(socket,
 *         asio::buffer(data, n), ctx);
 *
 *     bytes_transferred += n;
 *   }
 *   return bytes_transferred;
 * } *
 * // ...
 *
 * asio::async_fiber(my_context,
 *   [&](fiber_context ctx)
 *   {
 *      return echo(ctx, std::move(my_tcp_socket));
 *   },
 *   [](std::exception_ptr e, std::size_t n)
 *   {
 *     std::cout << "transferred " << n << "\n";
 *   });
 *
 * @endcode
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call asio::this_coro::reset_cancellation_state.
 */
template<typename ExecutionContext, typename Function,
        ASIO_COMPLETION_TOKEN_FOR(void(std::exception_ptr, decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))))
          CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(typename ExecutionContext::executor_type)>
auto async_fiber(ExecutionContext & ctx,
                 Function && func,
                 CompletionToken && completion_token,
                 typename constraint<
                         is_convertible<ExecutionContext&, execution_context&>::value
                         && !noexcept(func(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))
                         && !std::is_void_v<decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))>
                         >::type = 0)
{
  using sig_t = void (std::exception_ptr, decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>())));
  return async_initiate<CompletionToken, sig_t>
          (detail::initiate_fiber<typename ExecutionContext::executor_type>(ctx.get_executor()), completion_token,
                  std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}

/// Run a fiber based thread of execution.
/** This is an overload that can throw exception and return a value. The type is deduced from the function signature.
 *
 * @param exec The executor
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr, void)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket)
 * {
 *   std::size_t bytes_transferred = 0;
 *
 *   char data[1024];
 *   for (;;)
 *   {
 *     std::size_t n = socket.async_read_some(
 *         asio::buffer(data), ctx);
 *
 *     asio::async_write(socket,
 *         asio::buffer(data, n), ctx);
 *
 *     bytes_transferred += n;
 *   }
 *
 *   return bytes_transferred;
 * }
 *
 * // ...
 *
 * asio::async_fiber(my_executor,
 *   [&](fiber_context ctx)
 *   {
 *      return echo(ctx, std::move(my_tcp_socket));
 *   },
 *   [](std::exception_ptr e, std::size_t n)
 *   {
 *     std::cout << "transferred " << n << "\n";
 *   });
 *
 * @endcode
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call asio::this_coro::reset_cancellation_state.
 */
template<typename Executor, typename Function,
        ASIO_COMPLETION_TOKEN_FOR(void(std::exception_ptr,
                decltype(std::declval<Function>()(std::declval<basic_fiber_context<std::decay_t<Executor>>>()))))
          CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
auto async_fiber(const Executor & exec,
                 Function && func,
                 CompletionToken && completion_token ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
                 typename constraint<
                         ( is_executor<Executor>::value ||
                           execution::is_executor<Executor>::value)
                       && !noexcept(func(std::declval<basic_fiber_context<std::decay_t<Executor>>>()))
                       && !std::is_void_v<decltype(std::declval<Function>()(std::declval<basic_fiber_context<Executor>>()))>
                 >::type = 0)
{
  using sig_t = void (std::exception_ptr, decltype(std::declval<Function>()(std::declval<basic_fiber_context<std::decay_t<Executor>>>())));
  return async_initiate<CompletionToken, sig_t >(
          detail::initiate_fiber<Executor>(exec), completion_token, std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}

/// Run a fiber based thread of execution.
/** This is an overload that cannot throw exception and return a value. The type is deduced from the function signature
 * and the exception behaviour with `noexcept`.
 *
 * @param ctx The execution context
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr, void)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket) noexcept
 * {
 *   std::size_t bytes_transferred = 0;
 *
 *   try
 *   {
 *     char data[1024];
 *     for (;;)
 *     {
 *       std::size_t n = socket.async_read_some(
 *           asio::buffer(data), ctx);
 *
 *       asio::async_write(socket,
 *           asio::buffer(data, n), ctx);
 *
 *       bytes_transferred += n;
 *     }
 *   }
 *   catch (const std::exception&)
 *   {
 *   }
 *
 *   return bytes_transferred;
 * } *
 * // ...
 *
 * asio::async_fiber(my_context,
 *   [&](fiber_context ctx)
 *   {
 *      return echo(ctx, std::move(my_tcp_socket));
 *   },
 *   [](std::size_t n)
 *   {
 *     std::cout << "transferred " << n << "\n";
 *   });
 *
 * @endcode
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call asio::this_coro::reset_cancellation_state.
 */
template<typename ExecutionContext, typename Function,
        ASIO_COMPLETION_TOKEN_FOR(void(decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))))
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(typename ExecutionContext::executor_type)>
auto async_fiber(ExecutionContext & ctx,
                 Function && func,
                 CompletionToken && completion_token,
                 typename constraint<
                         is_convertible<ExecutionContext&, execution_context&>::value
                         && noexcept(func(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))
                         && !std::is_void_v<decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))>
                 >::type = 0)
{
  using sig_t = void (decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>())));
  return async_initiate<CompletionToken, sig_t>(
          detail::initiate_fiber<typename ExecutionContext::executor_type>(ctx.get_executor()), completion_token, std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}


/// Run a fiber based thread of execution.
/** This is an overload that cannot throw exception and return a value. The type is deduced from the function signature
 * and the exception behaviour with `noexcept`.
 *
 * @param exec The executor
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr, void)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket) noexcept
 * {
 *   std::size_t bytes_transferred = 0;
 *
 *   try
 *   {
 *     char data[1024];
 *     for (;;)
 *     {
 *       std::size_t n = socket.async_read_some(
 *           asio::buffer(data), ctx);
 *
 *       asio::async_write(socket,
 *           asio::buffer(data, n), ctx);
 *
 *       bytes_transferred += n;
 *     }
 *   }
 *   catch (const std::exception&)
 *   {
 *   }
 *
 *   return bytes_transferred;
 * } *
 * // ...
 *
 * asio::async_fiber(my_executor,
 *   [&](fiber_context ctx)
 *   {
 *      return echo(ctx, std::move(my_tcp_socket));
 *   },
 *   [](std::size_t n)
 *   {
 *     std::cout << "transferred " << n << "\n";
 *   });
 *
 * @endcode
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call asio::this_coro::reset_cancellation_state.
 */

template<typename Executor, typename Function,
        ASIO_COMPLETION_TOKEN_FOR(void(decltype(std::declval<Function>()(std::declval<basic_fiber_context<std::decay_t<Executor>>>()))))
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
auto async_fiber(const Executor & exec,
                 Function && func,
                 CompletionToken && completion_token ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
                 typename constraint<
                         ( is_executor<Executor>::value ||
                           execution::is_executor<Executor>::value)
                         && noexcept(func(std::declval<basic_fiber_context<std::decay_t<Executor>>>()))
                         && !std::is_void_v<decltype(std::declval<Function>()(std::declval<basic_fiber_context<Executor>>()))>
                 >::type = 0)
{
  using sig_t = void (decltype(std::declval<Function>()(std::declval<basic_fiber_context<std::decay_t<Executor>>>())));
  return async_initiate<CompletionToken, sig_t>(
          detail::initiate_fiber<Executor>(exec), completion_token, std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}

/// Run a fiber based thread of execution.
/** This is an overload that can throw exception and does not return a value.
 * The type is deduced from the function signature.
 *
 * @param ctx The execution context
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr, void)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket)
 * {
 *   std::size_t bytes_transferred = 0;
 *
 *   char data[1024];
 *   for (;;)
 *   {
 *     std::size_t n = socket.async_read_some(
 *         asio::buffer(data), ctx);
 *
 *     asio::async_write(socket,
 *         asio::buffer(data, n), ctx);
 *
 *     bytes_transferred += n;
 *   }
 * } *
 * // ...
 *
 * asio::async_fiber(my_context,
 *   [&](fiber_context ctx)
 *   {
 *      return echo(ctx, std::move(my_tcp_socket));
 *   },
 *   [](std::exception_ptr e)
 *   {
 *   });
 *
 * @endcode
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call asio::this_coro::reset_cancellation_state.
 */
template<typename ExecutionContext, typename Function,
        ASIO_COMPLETION_TOKEN_FOR(void(std::exception_ptr))
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(typename ExecutionContext::executor_type)>
auto async_fiber(ExecutionContext & ctx,
                 Function && func,
                 CompletionToken && completion_token,
                 typename constraint<
                         is_convertible<ExecutionContext&, execution_context&>::value
                         && !noexcept(func(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))
                         && std::is_void_v<decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))>
                 >::type = 0)
{
  using sig_t = void (std::exception_ptr);
  return async_initiate<CompletionToken, sig_t>
          (detail::initiate_fiber<typename ExecutionContext::executor_type>(ctx.get_executor()), completion_token,
           std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}


/// Run a fiber based thread of execution.
/** This is an overload that can throw exception and does not return a value.
 * The type is deduced from the function signature.
 *
 * @param exec The executor
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr, void)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket)
 * {
 *   std::size_t bytes_transferred = 0;
 *
 *   char data[1024];
 *   for (;;)
 *   {
 *     std::size_t n = socket.async_read_some(
 *         asio::buffer(data), ctx);
 *
 *     asio::async_write(socket,
 *         asio::buffer(data, n), ctx);
 *
 *     bytes_transferred += n;
 *   }
 *
 * }
 *
 * // ...
 *
 * asio::async_fiber(my_executor,
 *   [&](fiber_context ctx)
 *   {
 *      return echo(ctx, std::move(my_tcp_socket));
 *   },
 *   [](std::exception_ptr e)
 *   {
 *   });
 *
 * @endcode
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call asio::this_coro::reset_cancellation_state.
 */
template<typename Executor, typename Function,
        ASIO_COMPLETION_TOKEN_FOR(void(std::exception_ptr))
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
auto async_fiber(const Executor & exec,
                 Function && func,
                 CompletionToken && completion_token ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
                 typename constraint<
                         ( is_executor<Executor>::value ||
                           execution::is_executor<Executor>::value)
                         && !noexcept(func(std::declval<basic_fiber_context<std::decay_t<Executor>>>()))
                         && std::is_void_v<decltype(std::declval<Function>()(std::declval<basic_fiber_context<Executor>>()))>
                 >::type = 0)
{
  using sig_t = void (std::exception_ptr);
  return async_initiate<CompletionToken, sig_t>(
          detail::initiate_fiber<Executor>(exec), completion_token, std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}


/// Run a fiber based thread of execution.
/** This is an overload that cannot throw exception and does not return a value.
 * The type is deduced from the function signature
 * and the exception behaviour with `noexcept`.
 *
 * @param ctx The execution context
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr, void)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket) noexcept
 * {
 *   std::size_t bytes_transferred = 0;
 *
 *   try
 *   {
 *     char data[1024];
 *     for (;;)
 *     {
 *       std::size_t n = socket.async_read_some(
 *           asio::buffer(data), ctx);
 *
 *       asio::async_write(socket,
 *           asio::buffer(data, n), ctx);
 *
 *       bytes_transferred += n;
 *     }
 *   }
 *   catch (const std::exception&)
 *   {
 *   }
 *
 * } *
 * // ...
 *
 * asio::async_fiber(my_context,
 *   [&](fiber_context ctx)
 *   {
 *      return echo(ctx, std::move(my_tcp_socket));
 *   },
 *   []()
 *   {
 *   });
 *
 * @endcode
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call asio::this_coro::reset_cancellation_state.
 */
template<typename ExecutionContext, typename Function,
        ASIO_COMPLETION_TOKEN_FOR(void())
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(typename ExecutionContext::executor_type)>
auto async_fiber(ExecutionContext & ctx,
                 Function && func,
                 CompletionToken && completion_token,
                 typename constraint<
                         is_convertible<ExecutionContext&, execution_context&>::value
                         && noexcept(func(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))
                         && std::is_void_v<decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))>
                 >::type = 0)
{
  using sig_t = void ();
  return async_initiate<CompletionToken,
          sig_t>(
          detail::initiate_fiber<typename ExecutionContext::executor_type>(ctx.get_executor()), completion_token,
          std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}


/// Run a fiber based thread of execution.
/** This is an overload that cannot throw exception and does not return a value.
 * The type is deduced from the function signature
 * and the exception behaviour with `noexcept`.
 *
 * @param exec The executor
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr, void)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket) noexcept
 * {
 *   std::size_t bytes_transferred = 0;
 *
 *   try
 *   {
 *     char data[1024];
 *     for (;;)
 *     {
 *       std::size_t n = socket.async_read_some(
 *           asio::buffer(data), ctx);
 *
 *       asio::async_write(socket,
 *           asio::buffer(data, n), ctx);
 *
 *       bytes_transferred += n;
 *     }
 *   }
 *   catch (const std::exception&)
 *   {
 *   }
 *
 *   return bytes_transferred;
 * } *
 * // ...
 *
 * asio::async_fiber(my_executor,
 *   [&](fiber_context ctx)
 *   {
 *      return echo(ctx, std::move(my_tcp_socket));
 *   },
 *   []()
 *   {
 *   });
 *
 * @endcode
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call asio::this_coro::reset_cancellation_state.
 */
template<typename Executor, typename Function,
        ASIO_COMPLETION_TOKEN_FOR(void())
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
auto async_fiber(const Executor & exec,
                 Function && func,
                 CompletionToken && completion_token ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
                 typename constraint<
                         ( is_executor<Executor>::value ||
                           execution::is_executor<Executor>::value)
                         && noexcept(func(std::declval<basic_fiber_context<std::decay_t<Executor>>>()))
                         && std::is_void_v<decltype(std::declval<Function>()(std::declval<basic_fiber_context<Executor>>()))>
                 >::type = 0)
{
  using sig_t = void();
  return async_initiate<CompletionToken, sig_t>(
          detail::initiate_fiber<Executor>(exec), completion_token, std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}


#if !defined(GENERATING_DOCUMENTATION)

template <typename Executor, typename R>
struct async_result<basic_fiber_context<Executor>, R()>
{
  using return_type = void;
  template <typename Initiation, typename... InitArgs>
  static return_type initiate(Initiation initiation, basic_fiber_context<Executor> ctx , InitArgs... args)
  {
    std::move(initiation)(
            bind_cancellation_slot(
              ctx.handle_->cancel_state.slot(),
              bind_executor(
                ctx.executor_,
                [&ctx , lp = std::move(ctx.handle_->lifetime)] () mutable
                {
                  auto p = lp.get();
                  ctx.handle_->lifetime = std::move(lp);
                  p->fiber = std::move(p->fiber).resume();
                })), std::forward<InitArgs>(args)...);
    *ctx.fiber_ = std::move(*ctx.fiber_).resume(); // suspend myself
  }
};


template <typename Executor, typename R>
struct async_result<basic_fiber_context<Executor>, R(std::exception_ptr)>
{
  using return_type = void;
  template <typename Initiation, typename... InitArgs>
  static return_type initiate(Initiation initiation, basic_fiber_context<Executor> ctx , InitArgs... args)
  {
    std::exception_ptr e;
    std::move(initiation)(
            bind_cancellation_slot(
                    ctx.handle_->cancel_state.slot(),
                    bind_executor(
                            ctx.executor_,
                            [& , lp = std::move(ctx.handle_->lifetime)] (std::exception_ptr ep) mutable
                            {
                              auto p = lp.get();
                              ctx.handle_->lifetime = std::move(lp);
                              e = ep;
                              p->fiber = std::move(p->fiber).resume();

                            })), std::forward<InitArgs>(args)...);


    *ctx.fiber_ = std::move(*ctx.fiber_).resume(); // suspend myself

    if (e)
      std::rethrow_exception(e);
  }
};


template <typename Executor, typename R>
struct async_result<basic_fiber_context<Executor>, R(error_code)>
{
  using return_type = void;
  using completion_handler_type = basic_fiber_context<Executor>;
  template <typename Initiation, typename... InitArgs>
  static return_type initiate(Initiation initiation, basic_fiber_context<Executor> ctx , InitArgs... args)
  {
    asio::error_code ec;
    std::move(initiation)(
            bind_cancellation_slot(
                    ctx.handle_->cancel_state.slot(),
                    bind_executor(
                            ctx.executor_,
                            [& , lp = std::move(ctx.handle_->lifetime)] (asio::error_code e) mutable
                            {
                              auto p = lp.get();
                              ctx.handle_->lifetime = std::move(lp);
                              ec = e;
                              p->fiber = std::move(p->fiber).resume();

                            })), std::forward<InitArgs>(args)...);

    *ctx.fiber_ = std::move(*ctx.fiber_).resume(); // suspend myself

    if (ec)
      detail::throw_error(ec);
  }
};

template <typename Executor, typename R, typename T>
struct async_result<basic_fiber_context<Executor>, R(T)>
{
  using return_type = T;

  template <typename Initiation, typename... InitArgs>
  static return_type initiate(Initiation initiation, basic_fiber_context<Executor> ctx , InitArgs... args)
  {
    std::optional<T> res;

    std::move(initiation)(
            bind_cancellation_slot(
                    ctx.handle_->cancel_state.slot(),
                    bind_executor(
                            ctx.executor_,
                            [& , lp = std::move(ctx.handle_->lifetime)] (T r) mutable
                            {
                              auto p = lp.get();
                              ctx.handle_->lifetime = std::move(lp);
                              res.emplace(std::move(r));
                              p->fiber = std::move(p->fiber).resume();

                            })), std::forward<InitArgs>(args)...);

    *ctx.fiber_ = std::move(*ctx.fiber_).resume(); // suspend myself
    return std::move(*res);
  }
};

template <typename Executor, typename R, typename T>
struct async_result<basic_fiber_context<Executor>, R(std::exception_ptr, T)>
{
  using return_type = T;
  template <typename Initiation, typename... InitArgs>
  static return_type initiate(Initiation initiation, basic_fiber_context<Executor> ctx , InitArgs... args)
  {
    std::optional<T> res;
    std::exception_ptr e;

    std::move(initiation)(
            bind_cancellation_slot(
                    ctx.handle_->cancel_state.slot(),
                    bind_executor(
                            ctx.executor_,
                            [& , lp = std::move(ctx.handle_->lifetime)] (std::exception_ptr ep, T r) mutable
                            {
                              auto p = lp.get();
                              ctx.handle_->lifetime = std::move(lp);
                              if (ep)
                                e = ep;
                              else
                                res.emplace(std::move(r));

                              p->fiber = std::move(p->fiber).resume();

                            })), std::forward<InitArgs>(args)...);

    *ctx.fiber_ = std::move(*ctx.fiber_).resume(); // suspend myself

    if (e)
      std::rethrow_exception(e);
    return std::move(*res);
  }
};


template <typename Executor, typename R, typename T>
struct async_result<basic_fiber_context<Executor>, R(error_code, T)>
{
  using return_type = T;

  template <typename Initiation, typename... InitArgs>
  static return_type initiate(Initiation initiation, basic_fiber_context<Executor> ctx , InitArgs... args)
  {
    std::optional<T> res;
    asio::error_code ec;

    std::move(initiation)(
            bind_cancellation_slot(
                    ctx.handle_->cancel_state.slot(),
                    bind_executor(
                            ctx.executor_,
                            [& , lp = std::move(ctx.handle_->lifetime)] (error_code e, T r) mutable
                            {
                              auto p = lp.get();
                              ctx.handle_->lifetime = std::move(lp);
                              if (e)
                                ec = e;
                              else
                                res.emplace(std::move(r));

                              p->fiber = std::move(p->fiber).resume();

                            })), std::forward<InitArgs>(args)...);

    *ctx.fiber_ = std::move(*ctx.fiber_).resume(); // suspend myself

    if (ec)
      detail::throw_error(ec);
    return std::move(*res);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)


} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif //ASIO_FIBER_HPP
