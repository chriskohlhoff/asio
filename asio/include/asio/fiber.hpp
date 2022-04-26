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
#include "asio/this_coro.hpp"

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

  bool * completed = nullptr;

  /* The reason we need to point both ways is the
   * completion by dispatch or post.
   */
  void resume()
  {
    struct completer_t
    {
      fiber_handle & fh;
      bool done = false;

      completer_t(fiber_handle & fh)  : fh(fh)
      {
        fh.completed = & done;
      }
      ~completer_t()
      {
        if (!done)
          fh.completed = nullptr;
      }
    } completer{*this};
    did_suspend = true;
    auto tmp = std::move(fiber).resume();
    if (!completer.done)
      fiber = std::move(tmp);
  }

  fiber_handle(cancellation_slot  cancel_slot) : cancel_slot(std::move(cancel_slot))
  {
  }
  fiber_handle(const fiber_handle & ) = delete;
  ~fiber_handle()
  {
    if (completed)
      *completed = true;
  }
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
          typename constraint<std::is_convertible<Executor2, Executor>::value>::type = 0) noexcept
          : handle_(other.handle_), fiber_(other.fiber_), executor_(other.get_executor())
  {
  }

  //// move construct a basic_fiber_context from anoter context. Can convert the executor type.
  template<typename Executor2>
  basic_fiber_context(
          basic_fiber_context<Executor2>  && other,
          typename constraint<std::is_convertible<Executor2, Executor>::value>::type = 0) noexcept
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

 private:
  template<typename Executor2>
  friend class basic_fiber_context;

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

      void complete()
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
        func(fiber_context{*this, std::move(fb), executor}, std::nothrow);
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
        complete(func(fiber_context{*this, std::move(fb), executor}, std::nothrow));
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
        catch (boost::context::detail::forced_unwind &) {throw; }
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
        catch (boost::context::detail::forced_unwind &) {throw; }
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
 * @code void (std::exception_ptr, T)
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
                         && !std::is_void<decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))>::value
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
 * @code void (std::exception_ptr, T)
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
                decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename std::decay<Executor>::type>>()))))
          CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
auto async_fiber(const Executor & exec,
                 Function && func,
                 CompletionToken && completion_token ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
                 typename constraint<
                         ( is_executor<Executor>::value ||
                           execution::is_executor<Executor>::value)
                       && !noexcept(func(std::declval<basic_fiber_context<typename std::decay<Executor>::type>>()))
                       && !std::is_void<decltype(std::declval<Function>()(std::declval<basic_fiber_context<Executor>>()))>::value
                 >::type = 0)
{
  using sig_t = void (std::exception_ptr, decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename std::decay<Executor>::type>>())));
  return async_initiate<CompletionToken, sig_t >(
          detail::initiate_fiber<Executor>(exec), completion_token, std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}

/// Run a fiber based thread of execution.
/** This is an overload that cannot throw an exception and returns a value. The type and exception behaviour
 * are deduced from the function signature.
 *
 * @param ctx The execution context
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (T)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket)
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
 *   catch (const std::exception&) // let boost::context::detail::force_unwind pass through!
 *   {
 *   }
 *
 *   return bytes_transferred;
 * } *
 * // ...
 *
 * asio::async_fiber(my_context,
 *   [&](fiber_context ctx, std::nothrow_t)
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
        ASIO_COMPLETION_TOKEN_FOR(void(decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>(), std::nothrow))))
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(typename ExecutionContext::executor_type)>
auto async_fiber(ExecutionContext & ctx,
                 Function && func,
                 CompletionToken && completion_token,
                 typename constraint<
                         is_convertible<ExecutionContext&, execution_context&>::value
                         && !noexcept(func(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>(), std::nothrow))
                         && !std::is_void<decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>(), std::nothrow))>::value
                 >::type = 0)
{
  using sig_t = void (decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>(), std::nothrow)));
  return async_initiate<CompletionToken, sig_t>(
          detail::initiate_fiber<typename ExecutionContext::executor_type>(ctx.get_executor()), completion_token, std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}


/// Run a fiber based thread of execution.
/** This is an overload that cannot throw exception and returns a value. The type is deduced from the function signature
 * and the exception behaviour with `noexcept`.
 *
 * @param exec The executor
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (void)
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket)
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
 *   [&](fiber_context ctx, std::nothrow_t)
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
        ASIO_COMPLETION_TOKEN_FOR(void(decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename std::decay<Executor>::type>>(), std::nothrow))))
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
auto async_fiber(const Executor & exec,
                 Function && func,
                 CompletionToken && completion_token ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
                 typename constraint<
                         ( is_executor<Executor>::value ||
                           execution::is_executor<Executor>::value)
                         && !noexcept(func(std::declval<basic_fiber_context<typename std::decay<Executor>::type>>(), std::nothrow))
                         && !std::is_void<decltype(std::declval<Function>()(std::declval<basic_fiber_context<Executor>>(), std::nothrow))>::value
                 >::type = 0)
{
  using sig_t = void (decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename std::decay<Executor>::type>>(), std::nothrow)));
  return async_initiate<CompletionToken, sig_t>(
          detail::initiate_fiber<Executor>(exec), completion_token, std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}

/// Run a fiber based thread of execution.
/** This is an overload that can throw and exception and does not return a value.
 * The type is deduced from the function signature.
 *
 * @param ctx The execution context
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr)
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
                         && std::is_void<decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>()))>::value
                 >::type = 0)
{
  using sig_t = void (std::exception_ptr);
  return async_initiate<CompletionToken, sig_t>
          (detail::initiate_fiber<typename ExecutionContext::executor_type>(ctx.get_executor()), completion_token,
           std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}


/// Run a fiber based thread of execution.
/** This is an overload that can throw an exception and does not return a value.
 * The type is deduced from the function signature.
 *
 * @param exec The executor
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void (std::exception_ptr)
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
                         && !noexcept(func(std::declval<basic_fiber_context<typename std::decay<Executor>::type>>()))
                         && std::is_void<decltype(std::declval<Function>()(std::declval<basic_fiber_context<Executor>>()))>::value
                 >::type = 0)
{
  using sig_t = void (std::exception_ptr);
  return async_initiate<CompletionToken, sig_t>(
          detail::initiate_fiber<Executor>(exec), completion_token, std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}


/// Run a fiber based thread of execution.
/** This is an overload that cannot throw an exception and does not return a value.
 * The type is deduced from the function signature
 * and the exception behaviour with `noexcept`.
 *
 * @param ctx The execution context
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void ()
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket)
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
 *   [&](fiber_context ctx, std::nothrow_t)
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
                         && !noexcept(func(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>(), std::nothrow))
                         && std::is_void<decltype(std::declval<Function>()(std::declval<basic_fiber_context<typename ExecutionContext::executor_type>>(), std::nothrow))>::value
                 >::type = 0)
{
  using sig_t = void ();
  return async_initiate<CompletionToken,
          sig_t>(
          detail::initiate_fiber<typename ExecutionContext::executor_type>(ctx.get_executor()), completion_token,
          std::forward<Function>(func), static_cast<sig_t*>(nullptr));
}


/// Run a fiber based thread of execution.
/** This is an overload that cannot throw an exception and does not return a value.
 * The type and exception behaviour are deduced from the function signature.
 *
 * @param exec The executor
 * @param func The fiber function. Must accept a fiber_context as first argument.
 * @param completion_token The completion token to be invoked once the fiber is done.
 * @par Completion Signature
 *
 * @par Completion Signature
 * @code void ()
 *
 * @par Example
 * @code
 * std::size_t echo(fiber_context ctx, tcp::socket socket)
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
                         && !noexcept(func(std::declval<basic_fiber_context<typename std::decay<Executor>::type>>(), std::nothrow))
                         && std::is_void<decltype(std::declval<Function>()(std::declval<basic_fiber_context<Executor>>(), std::nothrow))>::value
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
                  p->resume();
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
                              p->resume();

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
                              p->resume();
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
    struct my_opt
    {
      alignas(T) unsigned char result[sizeof(T)];
      bool has_result = false;
      void emplace(T && res)
      {
        new (&result) T(std::move(res));
        has_result = true;
      }

      T & operator *() {return *reinterpret_cast<T*>(&result); }

      ~my_opt(){
        if (has_result)
          reinterpret_cast<T*>(&result)->~T();
      }
    } res;

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
                              p->resume();
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
    struct my_opt
    {
      alignas(T) unsigned char result[sizeof(T)];
      bool has_result = false;
      void emplace(T && res)
      {
        new (&result) T(std::move(res));
        has_result = true;
      }

      T & operator *() {return *static_cast<T*>(&result); }

      ~my_opt(){
        if (has_result)
          static_cast<T*>(&result)->~T();
      }
    } res;
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

                              p->resume();
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
    struct my_opt
    {
      alignas(T) unsigned char result[sizeof(T)];
      bool has_result = false;
      void emplace(T && res)
      {
        new (&result) T(std::move(res));
        has_result = true;
      }

      T & operator *() {return *static_cast<T*>(&result); }

      ~my_opt(){
        if (has_result)
          static_cast<T*>(&result)->~T();
      }
    } res;
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

                              p->resume();
                            })), std::forward<InitArgs>(args)...);

    *ctx.fiber_ = std::move(*ctx.fiber_).resume(); // suspend myself

    if (ec)
      detail::throw_error(ec);
    return std::move(*res);
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)


} // namespace asio

// pseudo coroutine
#if defined(ASIO_HAS_CO_AWAIT)
#if defined(ASIO_HAS_STD_COROUTINE)
#include <coroutine>
#else // defined(ASIO_HAS_STD_COROUTINE)
#include <experimental/coroutine>
#endif // defined(ASIO_HAS_STD_COROUTINE)

namespace asio::detail
{
#if defined(ASIO_HAS_STD_COROUTINE)
using std::suspend_never;
using std::coroutine_handle;
#else
using std::experimental::suspend_never;
using std::experimental::coroutine_handle;
#endif

template<typename Executor>
struct basic_fiber_promise_base
{
  basic_fiber_context<Executor> ctx;

  template<typename ... Args>
  basic_fiber_promise_base(const basic_fiber_context<Executor> & ctx, Args && ...) : ctx(ctx) {}

  template<typename Lambda, typename ... Args>
  basic_fiber_promise_base(Lambda &&, const basic_fiber_context<Executor> & ctx, Args && ...) : ctx(ctx) {}

  constexpr static suspend_never initial_suspend() noexcept {return {};}
  constexpr static suspend_never   final_suspend() noexcept {return {};}

  auto await_transform(this_coro::executor_t) noexcept
  {
    struct result
    {
      basic_fiber_context<Executor>* this_;

      bool await_ready() const noexcept
      {
        return true;
      }

      void await_suspend(coroutine_handle<void>) noexcept
      {
      }

      auto await_resume() const noexcept
      {
        return this_->get_executor();
      }
    };

    return result{&this->ctx};
  }


  // This await transformation obtains the associated cancellation state of the
  // thread of execution.
  auto await_transform(this_coro::cancellation_state_t) noexcept
  {
    struct result
    {
      basic_fiber_context<Executor>* this_;

      bool await_ready() const noexcept
      {
        return true;
      }

      void await_suspend(coroutine_handle<void>) noexcept
      {
      }

      auto await_resume() const noexcept
      {
        return this_->get_cancellation_state();
      }
    };

    return result{&this->ctx};
  }

  // This await transformation resets the associated cancellation state.
  auto await_transform(this_coro::reset_cancellation_state_0_t) noexcept
  {
    struct result
    {
      basic_fiber_context<Executor>* this_;

      bool await_ready() const noexcept
      {
        return true;
      }

      void await_suspend(coroutine_handle<void>) noexcept
      {
      }

      auto await_resume() const
      {
        return this_->reset_cancellation_state();
      }
    };

    return result{&this->ctx};
  }

  // This await transformation resets the associated cancellation state.
  template <typename Filter>
  auto await_transform(
          this_coro::reset_cancellation_state_1_t<Filter> reset) noexcept
  {
    struct result
    {
      basic_fiber_context<Executor>* this_;
      Filter filter_;

      bool await_ready() const noexcept
      {
        return true;
      }

      void await_suspend(coroutine_handle<void>) noexcept
      {
      }

      auto await_resume()
      {
        return this_->reset_cancellation_state(
                ASIO_MOVE_CAST(Filter)(filter_));
      }
    };

    return result{&this->ctx, ASIO_MOVE_CAST(Filter)(reset.filter)};
  }

  // This await transformation resets the associated cancellation state.
  template <typename InFilter, typename OutFilter>
  auto await_transform(
          this_coro::reset_cancellation_state_2_t<InFilter, OutFilter> reset)
  noexcept
  {
    struct result
    {
      basic_fiber_context<Executor>* this_;
      InFilter in_filter_;
      OutFilter out_filter_;

      bool await_ready() const noexcept
      {
        return true;
      }

      void await_suspend(coroutine_handle<void>) noexcept
      {
      }

      auto await_resume()
      {
        return this_->reset_cancellation_state(
                ASIO_MOVE_CAST(InFilter)(in_filter_),
                ASIO_MOVE_CAST(OutFilter)(out_filter_));
      }
    };

    return result{&this->ctx,
                  ASIO_MOVE_CAST(InFilter)(reset.in_filter),
                  ASIO_MOVE_CAST(OutFilter)(reset.out_filter)};
  }

  // This await transformation determines whether cancellation is propagated as
  // an exception.
  auto await_transform(this_coro::throw_if_cancelled_0_t)
  noexcept
  {
    struct result
    {
      basic_fiber_context<Executor>* this_;

      bool await_ready() const noexcept
      {
        return true;
      }

      void await_suspend(coroutine_handle<void>) noexcept
      {
      }

      auto await_resume()
      {
        return this_->throw_if_cancelled();
      }
    };

    return result{&this->ctx};
  }

  // This await transformation sets whether cancellation is propagated as an
  // exception.
  auto await_transform(this_coro::throw_if_cancelled_1_t throw_if_cancelled)
  noexcept
  {
    struct result
    {
      basic_fiber_context<Executor>* this_;
      bool value_;

      bool await_ready() const noexcept
      {
        return true;
      }

      void await_suspend(coroutine_handle<void>) noexcept
      {
      }

      auto await_resume()
      {
        this_->throw_if_cancelled(value_);
      }
    };

    return result{&this->ctx, throw_if_cancelled.value};
  }


  // This await transformation sets whether cancellation is propagated as an
  // exception.
  template<typename ... Args>
  auto await_transform(asio::experimental::deferred_async_operation<Args...> && op)
  noexcept
  {
    struct result
    {
      basic_fiber_context<Executor>* this_;
      asio::experimental::deferred_async_operation<Args...> && op;

      bool await_ready() const noexcept
      {
        return true;
      }

      void await_suspend(coroutine_handle<void>) noexcept
      {
      }

      auto await_resume()
      {
        op(*this_);
      }
    };

    return result{&this->ctx, std::move(op)};
  }

};

template<typename Return, typename Executor>
struct basic_fiber_promise : basic_fiber_promise_base<Executor>
{
  using basic_fiber_promise_base<Executor>::basic_fiber_promise_base;

  struct proxy
  {

    proxy(Return ** target) : indirection(target)
    {
      *indirection = &this->target;
    }

    proxy(const proxy & rhs) : indirection(rhs.indirection)
    {
      *indirection = target;
    }

    operator Return()
    {
      return std::move(target);
    }

    Return target, ** indirection;
  };

  // UGLY AF
  Return * target = nullptr;

  template<typename Return_>
  void return_value(Return_ && r)
  {
    assert(target != nullptr);
    *target = std::forward<Return_>(r);
  }
  proxy get_return_object()
  {
    return proxy{&target};
  }
  void unhandled_exception() { throw ;}

};


template<typename Executor>
struct basic_fiber_promise<void, Executor> : basic_fiber_promise_base<Executor>
{
  using basic_fiber_promise_base<Executor>::basic_fiber_promise_base;

  constexpr static void return_void()
  {
  }
  void get_return_object() {}
  void unhandled_exception() { throw ;}

  std::exception_ptr ex;
};

}

#if defined(ASIO_HAS_STD_COROUTINE)
namespace std
#else // defined(ASIO_HAS_STD_COROUTINE)
namespace std::experimental
#endif // defined(ASIO_HAS_STD_COROUTINE)
{

template<typename Return, typename Executor, typename ... Args>
struct coroutine_traits<Return, asio::basic_fiber_context<Executor>, Args...>
{
  using promise_type = asio::detail::basic_fiber_promise<Return, Executor>;
};

template<typename Return, typename Lambda, typename Executor, typename ... Args>
struct coroutine_traits<Return, Lambda, asio::basic_fiber_context<Executor> , Args...>
{
  using promise_type = asio::detail::basic_fiber_promise<Return, Executor>;
};

template<typename Return, typename Executor, typename ... Args>
struct coroutine_traits<Return, asio::basic_fiber_context<Executor> &&, Args...>
{
  using promise_type = asio::detail::basic_fiber_promise<Return, Executor>;
};

template<typename Return, typename Lambda, typename Executor, typename ... Args>
struct coroutine_traits<Return, Lambda, asio::basic_fiber_context<Executor> && , Args...>
{
  using promise_type = asio::detail::basic_fiber_promise<Return, Executor>;
};

template<typename Return, typename Executor, typename ... Args>
struct coroutine_traits<Return, asio::basic_fiber_context<Executor> &, Args...>
{
  using promise_type = asio::detail::basic_fiber_promise<Return, Executor>;
};

template<typename Return, typename Lambda, typename Executor, typename ... Args>
struct coroutine_traits<Return, Lambda, asio::basic_fiber_context<Executor> & , Args...>
{
  using promise_type = asio::detail::basic_fiber_promise<Return, Executor>;
};


}


#endif



#include "asio/detail/pop_options.hpp"

#endif //ASIO_FIBER_HPP
