//
// spawn.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SPAWN_HPP
#define ASIO_SPAWN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/any_io_executor.hpp"
#include "asio/cancellation_signal.hpp"
#include "asio/cancellation_state.hpp"
#include "asio/detail/exception.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/io_context.hpp"
#include "asio/is_executor.hpp"
#include "asio/strand.hpp"

#if defined(ASIO_HAS_BOOST_COROUTINE)
# include <boost/coroutine/all.hpp>
#endif // defined(ASIO_HAS_BOOST_COROUTINE)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Base class for all spawn()-ed thread implementations.
class spawned_thread_base
{
public:
  spawned_thread_base()
    : owner_(0),
      has_context_switched_(false),
      throw_if_cancelled_(false),
      terminal_(false)
  {
  }

  virtual ~spawned_thread_base() {}
  virtual void resume() = 0;
  virtual void suspend_with(void (*fn)(void*), void* arg) = 0;
  virtual void destroy() = 0;

  void attach(spawned_thread_base** owner)
  {
    owner_ = owner;
    *owner_ = this;
  }

  void detach()
  {
    if (owner_)
      *owner_ = 0;
    owner_ = 0;
  }

  void suspend()
  {
    suspend_with(0, 0);
  }

  template <typename F>
  void suspend_with(F f)
  {
    suspend_with(&spawned_thread_base::call<F>, &f);
  }

  cancellation_slot get_cancellation_slot() const ASIO_NOEXCEPT
  {
    return cancellation_state_.slot();
  }

  cancellation_state get_cancellation_state() const ASIO_NOEXCEPT
  {
    return cancellation_state_;
  }

  void reset_cancellation_state()
  {
    cancellation_state_ = cancellation_state(parent_cancellation_slot_);
  }

  template <typename Filter>
  void reset_cancellation_state(Filter filter)
  {
    cancellation_state_ = cancellation_state(
        parent_cancellation_slot_, filter, filter);
  }

  template <typename InFilter, typename OutFilter>
  void reset_cancellation_state(InFilter in_filter, OutFilter out_filter)
  {
    cancellation_state_ = cancellation_state(
        parent_cancellation_slot_, in_filter, out_filter);
  }

  cancellation_type_t cancelled() const ASIO_NOEXCEPT
  {
    return cancellation_state_.cancelled();
  }

  bool has_context_switched() const ASIO_NOEXCEPT
  {
    return has_context_switched_;
  }

  bool throw_if_cancelled() const ASIO_NOEXCEPT
  {
    return throw_if_cancelled_;
  }

  void throw_if_cancelled(bool value) ASIO_NOEXCEPT
  {
    throw_if_cancelled_ = value;
  }

protected:
  spawned_thread_base** owner_; // Points to data member in active handler.
  asio::cancellation_slot parent_cancellation_slot_;
  asio::cancellation_state cancellation_state_;
  bool has_context_switched_;
  bool throw_if_cancelled_;
  bool terminal_;

private:
  // Disallow copying and assignment.
  spawned_thread_base(const spawned_thread_base&) ASIO_DELETED;
  spawned_thread_base& operator=(const spawned_thread_base&) ASIO_DELETED;

  template <typename F>
  static void call(void* f)
  {
    (*static_cast<F*>(f))();
  }
};


template <typename T>
struct spawn_signature
{
  typedef void type(exception_ptr, T);
};

template <>
struct spawn_signature<void>
{
  typedef void type(exception_ptr);
};

template <typename Executor>
class initiate_spawn;

} // namespace detail

/// A @ref completion_token that represents the currently executing coroutine.
/**
 * The basic_yield_context class is a completion token type that is used to
 * represent the currently executing stackful coroutine. A basic_yield_context
 * object may be passed as a completion token to an asynchronous operation. For
 * example:
 *
 * @code template <typename Executor>
 * void my_coroutine(basic_yield_context<Executor> yield)
 * {
 *   ...
 *   std::size_t n = my_socket.async_read_some(buffer, yield);
 *   ...
 * } @endcode
 *
 * The initiating function (async_read_some in the above example) suspends the
 * current coroutine. The coroutine is resumed when the asynchronous operation
 * completes, and the result of the operation is returned.
 */
template <typename Executor>
class basic_yield_context
{
public:
  /// The executor type associated with the yield context.
  typedef Executor executor_type;

  /// The cancellation slot type associated with the yield context.
  typedef cancellation_slot cancellation_slot_type;

  /// Construct a yield context from another yield context type.
  /**
   * Requires that OtherExecutor be convertible to Executor.
   */
  template <typename OtherExecutor>
  basic_yield_context(const basic_yield_context<OtherExecutor>& other,
      typename constraint<
        is_convertible<OtherExecutor, Executor>::value
      >::type = 0)
    : spawned_thread_(other.spawned_thread_),
      executor_(other.executor_),
      ec_(other.ec_)
  {
  }

  /// Get the executor associated with the yield context.
  executor_type get_executor() const ASIO_NOEXCEPT
  {
    return executor_;
  }

  /// Get the cancellation slot associated with the coroutine.
  cancellation_slot_type get_cancellation_slot() const ASIO_NOEXCEPT
  {
    return spawned_thread_->get_cancellation_slot();
  }

  /// Get the cancellation state associated with the coroutine.
  cancellation_state get_cancellation_state() const ASIO_NOEXCEPT
  {
    return spawned_thread_->get_cancellation_state();
  }

  /// Reset the cancellation state associated with the coroutine.
  /**
   * Let <tt>P</tt> be the cancellation slot associated with the current
   * coroutine's @ref spawn completion handler. Assigns a new
   * asio::cancellation_state object <tt>S</tt>, constructed as
   * <tt>S(P)</tt>, into the current coroutine's cancellation state object.
   */
  void reset_cancellation_state() const
  {
    spawned_thread_->reset_cancellation_state();
  }

  /// Reset the cancellation state associated with the coroutine.
  /**
   * Let <tt>P</tt> be the cancellation slot associated with the current
   * coroutine's @ref spawn completion handler. Assigns a new
   * asio::cancellation_state object <tt>S</tt>, constructed as <tt>S(P,
   * std::forward<Filter>(filter))</tt>, into the current coroutine's
   * cancellation state object.
   */
  template <typename Filter>
  void reset_cancellation_state(ASIO_MOVE_ARG(Filter) filter) const
  {
    spawned_thread_->reset_cancellation_state(
        ASIO_MOVE_CAST(Filter)(filter));
  }

  /// Reset the cancellation state associated with the coroutine.
  /**
   * Let <tt>P</tt> be the cancellation slot associated with the current
   * coroutine's @ref spawn completion handler. Assigns a new
   * asio::cancellation_state object <tt>S</tt>, constructed as <tt>S(P,
   * std::forward<InFilter>(in_filter),
   * std::forward<OutFilter>(out_filter))</tt>, into the current coroutine's
   * cancellation state object.
   */
  template <typename InFilter, typename OutFilter>
  void reset_cancellation_state(ASIO_MOVE_ARG(InFilter) in_filter,
      ASIO_MOVE_ARG(OutFilter) out_filter) const
  {
    spawned_thread_->reset_cancellation_state(
        ASIO_MOVE_CAST(InFilter)(in_filter),
        ASIO_MOVE_CAST(OutFilter)(out_filter));
  }

  /// Determine whether the current coroutine has been cancelled.
  cancellation_type_t cancelled() const ASIO_NOEXCEPT
  {
    return spawned_thread_->cancelled();
  }

  /// Determine whether the coroutine throws if trying to suspend when it has
  /// been cancelled.
  bool throw_if_cancelled() const ASIO_NOEXCEPT
  {
    return spawned_thread_->throw_if_cancelled();
  }

  /// Set whether the coroutine throws if trying to suspend when it has been
  /// cancelled.
  void throw_if_cancelled(bool value) const ASIO_NOEXCEPT
  {
    spawned_thread_->throw_if_cancelled(value);
  }

  /// Return a yield context that sets the specified error_code.
  /**
   * By default, when a yield context is used with an asynchronous operation, a
   * non-success error_code is converted to system_error and thrown. This
   * operator may be used to specify an error_code object that should instead be
   * set with the asynchronous operation's result. For example:
   *
   * @code template <typename Executor>
   * void my_coroutine(basic_yield_context<Executor> yield)
   * {
   *   ...
   *   std::size_t n = my_socket.async_read_some(buffer, yield[ec]);
   *   if (ec)
   *   {
   *     // An error occurred.
   *   }
   *   ...
   * } @endcode
   */
  basic_yield_context operator[](asio::error_code& ec) const
  {
    basic_yield_context tmp(*this);
    tmp.ec_ = &ec;
    return tmp;
  }

#if !defined(GENERATING_DOCUMENTATION)
//private:
  basic_yield_context(detail::spawned_thread_base* spawned_thread,
      const Executor& ex)
    : spawned_thread_(spawned_thread),
      executor_(ex),
      ec_(0)
  {
  }

  detail::spawned_thread_base* spawned_thread_;
  Executor executor_;
  asio::error_code* ec_;
#endif // !defined(GENERATING_DOCUMENTATION)
};

/// A @ref completion_token object that represents the currently executing
/// coroutine.
typedef basic_yield_context<any_io_executor> yield_context;

/**
 * @defgroup spawn asio::spawn
 *
 * @brief Start a new stackful coroutine.
 *
 * The spawn() function is a high-level wrapper over the Boost.Coroutine
 * library. This function enables programs to implement asynchronous logic in a
 * synchronous manner, as illustrated by the following example:
 *
 * @code asio::spawn(my_strand, do_echo, asio::detached);
 *
 * // ...
 *
 * void do_echo(asio::yield_context yield)
 * {
 *   try
 *   {
 *     char data[128];
 *     for (;;)
 *     {
 *       std::size_t length =
 *         my_socket.async_read_some(
 *           asio::buffer(data), yield);
 *
 *       asio::async_write(my_socket,
 *           asio::buffer(data, length), yield);
 *     }
 *   }
 *   catch (std::exception& e)
 *   {
 *     // ...
 *   }
 * } @endcode
 */
/*@{*/

/// Start a new stackful coroutine that executes on a given executor.
/**
 * This function is used to launch a new stackful coroutine.
 *
 * @param ex Identifies the executor that will run the stackful coroutine.
 *
 * @param function The coroutine function. The function must be callable the
 * signature:
 * @code void function(basic_yield_context<Executor> yield); @endcode
 *
 * @param token The @ref completion_token that will handle the notification
 * that the coroutine has completed. If the return type @c R of @c function is
 * @c void, the function signature of the completion handler must be:
 *
 * @code void handler(std::exception_ptr); @endcode
 * Otherwise, the function signature of the completion handler must be:
 * @code void handler(std::exception_ptr, R); @endcode
 *
 * @par Completion Signature
 * @code void(std::exception_ptr, R) @endcode
 * where @c R is the return type of the function object.
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call the basic_yield_context member function
 * @c reset_cancellation_state.
 */
template <typename Executor, typename F,
    ASIO_COMPLETION_TOKEN_FOR(typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<Executor>)>::type>::type)
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(CompletionToken,
    typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<Executor>)>::type>::type)
spawn(const Executor& ex, ASIO_MOVE_ARG(F) function,
    ASIO_MOVE_ARG(CompletionToken) token
      ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
#if defined(ASIO_HAS_BOOST_COROUTINE)
    typename constraint<
      !is_same<
        typename decay<CompletionToken>::type,
        boost::coroutines::attributes
      >::value
    >::type = 0,
#endif // defined(ASIO_HAS_BOOST_COROUTINE)
    typename constraint<
      is_executor<Executor>::value || execution::is_executor<Executor>::value
    >::type = 0)
  ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX((
    async_initiate<CompletionToken,
      typename detail::spawn_signature<
        typename result_of<F(basic_yield_context<Executor>)>::type>::type>(
          declval<detail::initiate_spawn<Executor> >(),
          token, ASIO_MOVE_CAST(F)(function))));

/// Start a new stackful coroutine that executes on a given execution context.
/**
 * This function is used to launch a new stackful coroutine.
 *
 * @param ctx Identifies the execution context that will run the stackful
 * coroutine.
 *
 * @param function The coroutine function. The function must be callable the
 * signature:
 * @code void function(basic_yield_context<Executor> yield); @endcode
 *
 * @param token The @ref completion_token that will handle the notification
 * that the coroutine has completed. If the return type @c R of @c function is
 * @c void, the function signature of the completion handler must be:
 *
 * @code void handler(std::exception_ptr); @endcode
 * Otherwise, the function signature of the completion handler must be:
 * @code void handler(std::exception_ptr, R); @endcode
 *
 * @par Completion Signature
 * @code void(std::exception_ptr, R) @endcode
 * where @c R is the return type of the function object.
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call the basic_yield_context member function
 * @c reset_cancellation_state.
 */
template <typename ExecutionContext, typename F,
    ASIO_COMPLETION_TOKEN_FOR(typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<
        typename ExecutionContext::executor_type>)>::type>::type)
          CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(
            typename ExecutionContext::executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(CompletionToken,
    typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<
        typename ExecutionContext::executor_type>)>::type>::type)
spawn(ExecutionContext& ctx, ASIO_MOVE_ARG(F) function,
    ASIO_MOVE_ARG(CompletionToken) token
      ASIO_DEFAULT_COMPLETION_TOKEN(
        typename ExecutionContext::executor_type),
#if defined(ASIO_HAS_BOOST_COROUTINE)
    typename constraint<
      !is_same<
        typename decay<CompletionToken>::type,
        boost::coroutines::attributes
      >::value
    >::type = 0,
#endif // defined(ASIO_HAS_BOOST_COROUTINE)
    typename constraint<
      is_convertible<ExecutionContext&, execution_context&>::value
    >::type = 0)
  ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX((
    async_initiate<CompletionToken,
      typename detail::spawn_signature<
        typename result_of<F(basic_yield_context<
          typename ExecutionContext::executor_type>)>::type>::type>(
            declval<detail::initiate_spawn<
              typename ExecutionContext::executor_type> >(),
            token, ASIO_MOVE_CAST(F)(function))));

/// Start a new stackful coroutine, inheriting the executor of another.
/**
 * This function is used to launch a new stackful coroutine.
 *
 * @param ctx Identifies the current coroutine as a parent of the new
 * coroutine. This specifies that the new coroutine should inherit the executor
 * of the parent. For example, if the parent coroutine is executing in a
 * particular strand, then the new coroutine will execute in the same strand.
 *
 * @param function The coroutine function. The function must be callable the
 * signature:
 * @code void function(basic_yield_context<Executor> yield); @endcode
 *
 * @param token The @ref completion_token that will handle the notification
 * that the coroutine has completed. If the return type @c R of @c function is
 * @c void, the function signature of the completion handler must be:
 *
 * @code void handler(std::exception_ptr); @endcode
 * Otherwise, the function signature of the completion handler must be:
 * @code void handler(std::exception_ptr, R); @endcode
 *
 * @par Completion Signature
 * @code void(std::exception_ptr, R) @endcode
 * where @c R is the return type of the function object.
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call the basic_yield_context member function
 * @c reset_cancellation_state.
 */
template <typename Executor, typename F,
    ASIO_COMPLETION_TOKEN_FOR(typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<Executor>)>::type>::type)
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(CompletionToken,
    typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<Executor>)>::type>::type)
spawn(const basic_yield_context<Executor>& ctx,
    ASIO_MOVE_ARG(F) function,
    ASIO_MOVE_ARG(CompletionToken) token
      ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
#if defined(ASIO_HAS_BOOST_COROUTINE)
    typename constraint<
      !is_same<
        typename decay<CompletionToken>::type,
        boost::coroutines::attributes
      >::value
    >::type = 0,
#endif // defined(ASIO_HAS_BOOST_COROUTINE)
    typename constraint<
      is_executor<Executor>::value || execution::is_executor<Executor>::value
    >::type = 0)
  ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX((
    async_initiate<CompletionToken,
      typename detail::spawn_signature<
        typename result_of<F(basic_yield_context<Executor>)>::type>::type>(
          declval<detail::initiate_spawn<Executor> >(),
          token, ASIO_MOVE_CAST(F)(function))));

#if defined(ASIO_HAS_BOOST_CONTEXT_FIBER) \
  || defined(GENERATING_DOCUMENTATION)

/// Start a new stackful coroutine that executes on a given executor.
/**
 * This function is used to launch a new stackful coroutine using the
 * specified stack allocator.
 *
 * @param ex Identifies the executor that will run the stackful coroutine.
 *
 * @param stack_allocator Denotes the allocator to be used to allocate the
 * underlying coroutine's stack. The type must satisfy the stack-allocator
 * concept defined by the Boost.Context library.
 *
 * @param function The coroutine function. The function must be callable the
 * signature:
 * @code void function(basic_yield_context<Executor> yield); @endcode
 *
 * @param token The @ref completion_token that will handle the notification
 * that the coroutine has completed. If the return type @c R of @c function is
 * @c void, the function signature of the completion handler must be:
 *
 * @code void handler(std::exception_ptr); @endcode
 * Otherwise, the function signature of the completion handler must be:
 * @code void handler(std::exception_ptr, R); @endcode
 *
 * @par Completion Signature
 * @code void(std::exception_ptr, R) @endcode
 * where @c R is the return type of the function object.
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call the basic_yield_context member function
 * @c reset_cancellation_state.
 */
template <typename Executor, typename StackAllocator, typename F,
    ASIO_COMPLETION_TOKEN_FOR(typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<Executor>)>::type>::type)
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(CompletionToken,
    typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<Executor>)>::type>::type)
spawn(const Executor& ex, allocator_arg_t,
    ASIO_MOVE_ARG(StackAllocator) stack_allocator,
    ASIO_MOVE_ARG(F) function,
    ASIO_MOVE_ARG(CompletionToken) token
      ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
    typename constraint<
      is_executor<Executor>::value || execution::is_executor<Executor>::value
    >::type = 0)
  ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX((
    async_initiate<CompletionToken,
      typename detail::spawn_signature<
        typename result_of<F(basic_yield_context<Executor>)>::type>::type>(
          declval<detail::initiate_spawn<Executor> >(),
          token, allocator_arg_t(),
          ASIO_MOVE_CAST(StackAllocator)(stack_allocator),
          ASIO_MOVE_CAST(F)(function))));

/// Start a new stackful coroutine that executes on a given execution context.
/**
 * This function is used to launch a new stackful coroutine.
 *
 * @param ctx Identifies the execution context that will run the stackful
 * coroutine.
 *
 * @param stack_allocator Denotes the allocator to be used to allocate the
 * underlying coroutine's stack. The type must satisfy the stack-allocator
 * concept defined by the Boost.Context library.
 *
 * @param function The coroutine function. The function must be callable the
 * signature:
 * @code void function(basic_yield_context<Executor> yield); @endcode
 *
 * @param token The @ref completion_token that will handle the notification
 * that the coroutine has completed. If the return type @c R of @c function is
 * @c void, the function signature of the completion handler must be:
 *
 * @code void handler(std::exception_ptr); @endcode
 * Otherwise, the function signature of the completion handler must be:
 * @code void handler(std::exception_ptr, R); @endcode
 *
 * @par Completion Signature
 * @code void(std::exception_ptr, R) @endcode
 * where @c R is the return type of the function object.
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call the basic_yield_context member function
 * @c reset_cancellation_state.
 */
template <typename ExecutionContext, typename StackAllocator, typename F,
    ASIO_COMPLETION_TOKEN_FOR(typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<
        typename ExecutionContext::executor_type>)>::type>::type)
          CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(
            typename ExecutionContext::executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(CompletionToken,
    typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<
        typename ExecutionContext::executor_type>)>::type>::type)
spawn(ExecutionContext& ctx, allocator_arg_t,
    ASIO_MOVE_ARG(StackAllocator) stack_allocator,
    ASIO_MOVE_ARG(F) function,
    ASIO_MOVE_ARG(CompletionToken) token
      ASIO_DEFAULT_COMPLETION_TOKEN(
        typename ExecutionContext::executor_type),
    typename constraint<
      is_convertible<ExecutionContext&, execution_context&>::value
    >::type = 0)
  ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX((
    async_initiate<CompletionToken,
      typename detail::spawn_signature<
        typename result_of<F(basic_yield_context<
          typename ExecutionContext::executor_type>)>::type>::type>(
            declval<detail::initiate_spawn<
              typename ExecutionContext::executor_type> >(),
            token, allocator_arg_t(),
            ASIO_MOVE_CAST(StackAllocator)(stack_allocator),
            ASIO_MOVE_CAST(F)(function))));

/// Start a new stackful coroutine, inheriting the executor of another.
/**
 * This function is used to launch a new stackful coroutine using the
 * specified stack allocator.
 *
 * @param ctx Identifies the current coroutine as a parent of the new
 * coroutine. This specifies that the new coroutine should inherit the
 * executor of the parent. For example, if the parent coroutine is executing
 * in a particular strand, then the new coroutine will execute in the same
 * strand.
 *
 * @param stack_allocator Denotes the allocator to be used to allocate the
 * underlying coroutine's stack. The type must satisfy the stack-allocator
 * concept defined by the Boost.Context library.
 *
 * @param function The coroutine function. The function must be callable the
 * signature:
 * @code void function(basic_yield_context<Executor> yield); @endcode
 *
 * @param token The @ref completion_token that will handle the notification
 * that the coroutine has completed. If the return type @c R of @c function is
 * @c void, the function signature of the completion handler must be:
 *
 * @code void handler(std::exception_ptr); @endcode
 * Otherwise, the function signature of the completion handler must be:
 * @code void handler(std::exception_ptr, R); @endcode
 *
 * @par Completion Signature
 * @code void(std::exception_ptr, R) @endcode
 * where @c R is the return type of the function object.
 *
 * @par Per-Operation Cancellation
 * The new thread of execution is created with a cancellation state that
 * supports @c cancellation_type::terminal values only. To change the
 * cancellation state, call the basic_yield_context member function
 * @c reset_cancellation_state.
 */
template <typename Executor, typename StackAllocator, typename F,
    ASIO_COMPLETION_TOKEN_FOR(typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<Executor>)>::type>::type)
        CompletionToken ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(Executor)>
ASIO_INITFN_AUTO_RESULT_TYPE_PREFIX(CompletionToken,
    typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<Executor>)>::type>::type)
spawn(const basic_yield_context<Executor>& ctx, allocator_arg_t,
    ASIO_MOVE_ARG(StackAllocator) stack_allocator,
    ASIO_MOVE_ARG(F) function,
    ASIO_MOVE_ARG(CompletionToken) token
    ASIO_DEFAULT_COMPLETION_TOKEN(Executor),
  typename constraint<
    is_executor<Executor>::value || execution::is_executor<Executor>::value
  >::type = 0)
ASIO_INITFN_AUTO_RESULT_TYPE_SUFFIX((
  async_initiate<CompletionToken,
    typename detail::spawn_signature<
      typename result_of<F(basic_yield_context<Executor>)>::type>::type>(
        declval<detail::initiate_spawn<Executor> >(),
        token, allocator_arg_t(),
        ASIO_MOVE_CAST(StackAllocator)(stack_allocator),
        ASIO_MOVE_CAST(F)(function))));

#endif // defined(ASIO_HAS_BOOST_CONTEXT_FIBER)
       //   || defined(GENERATING_DOCUMENTATION)

#if defined(ASIO_HAS_BOOST_COROUTINE) \
  || defined(GENERATING_DOCUMENTATION)

/// (Deprecated: Use overloads with a completion token.) Start a new stackful
/// coroutine, calling the specified handler when it completes.
/**
 * This function is used to launch a new coroutine.
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(basic_yield_context<Executor> yield); @endcode
 * where Executor is the associated executor type of @c Function.
 *
 * @param attributes Boost.Coroutine attributes used to customise the coroutine.
 */
template <typename Function>
void spawn(ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes());

/// (Deprecated: Use overloads with a completion token.) Start a new stackful
/// coroutine, calling the specified handler when it completes.
/**
 * This function is used to launch a new coroutine.
 *
 * @param handler A handler to be called when the coroutine exits. More
 * importantly, the handler provides an execution context (via the the handler
 * invocation hook) for the coroutine. The handler must have the signature:
 * @code void handler(); @endcode
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(basic_yield_context<Executor> yield); @endcode
 * where Executor is the associated executor type of @c Handler.
 *
 * @param attributes Boost.Coroutine attributes used to customise the coroutine.
 */
template <typename Handler, typename Function>
void spawn(ASIO_MOVE_ARG(Handler) handler,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes(),
    typename constraint<
      !is_executor<typename decay<Handler>::type>::value &&
      !execution::is_executor<typename decay<Handler>::type>::value &&
      !is_convertible<Handler&, execution_context&>::value>::type = 0);

/// (Deprecated: Use overloads with a completion token.) Start a new stackful
/// coroutine, inheriting the execution context of another.
/**
 * This function is used to launch a new coroutine.
 *
 * @param ctx Identifies the current coroutine as a parent of the new
 * coroutine. This specifies that the new coroutine should inherit the
 * execution context of the parent. For example, if the parent coroutine is
 * executing in a particular strand, then the new coroutine will execute in the
 * same strand.
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(basic_yield_context<Executor> yield); @endcode
 *
 * @param attributes Boost.Coroutine attributes used to customise the coroutine.
 */
template <typename Executor, typename Function>
void spawn(basic_yield_context<Executor> ctx,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes());

/// (Deprecated: Use overloads with a completion token.) Start a new stackful
/// coroutine that executes on a given executor.
/**
 * This function is used to launch a new coroutine.
 *
 * @param ex Identifies the executor that will run the coroutine. The new
 * coroutine is automatically given its own explicit strand within this
 * executor.
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(yield_context yield); @endcode
 *
 * @param attributes Boost.Coroutine attributes used to customise the coroutine.
 */
template <typename Function, typename Executor>
void spawn(const Executor& ex,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes(),
    typename constraint<
      is_executor<Executor>::value || execution::is_executor<Executor>::value
    >::type = 0);

/// (Deprecated: Use overloads with a completion token.) Start a new stackful
/// coroutine that executes on a given strand.
/**
 * This function is used to launch a new coroutine.
 *
 * @param ex Identifies the strand that will run the coroutine.
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(yield_context yield); @endcode
 *
 * @param attributes Boost.Coroutine attributes used to customise the coroutine.
 */
template <typename Function, typename Executor>
void spawn(const strand<Executor>& ex,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes());

#if !defined(ASIO_NO_TS_EXECUTORS)

/// (Deprecated: Use overloads with a completion token.) Start a new stackful
/// coroutine that executes in the context of a strand.
/**
 * This function is used to launch a new coroutine.
 *
 * @param s Identifies a strand. By starting multiple coroutines on the same
 * strand, the implementation ensures that none of those coroutines can execute
 * simultaneously.
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(yield_context yield); @endcode
 *
 * @param attributes Boost.Coroutine attributes used to customise the coroutine.
 */
template <typename Function>
void spawn(const asio::io_context::strand& s,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes());

#endif // !defined(ASIO_NO_TS_EXECUTORS)

/// (Deprecated: Use overloads with a completion token.) Start a new stackful
/// coroutine that executes on a given execution context.
/**
 * This function is used to launch a new coroutine.
 *
 * @param ctx Identifies the execution context that will run the coroutine. The
 * new coroutine is implicitly given its own strand within this execution
 * context.
 *
 * @param function The coroutine function. The function must have the signature:
 * @code void function(yield_context yield); @endcode
 *
 * @param attributes Boost.Coroutine attributes used to customise the coroutine.
 */
template <typename Function, typename ExecutionContext>
void spawn(ExecutionContext& ctx,
    ASIO_MOVE_ARG(Function) function,
    const boost::coroutines::attributes& attributes
      = boost::coroutines::attributes(),
    typename constraint<is_convertible<
      ExecutionContext&, execution_context&>::value>::type = 0);

#endif // defined(ASIO_HAS_BOOST_COROUTINE)
       //   || defined(GENERATING_DOCUMENTATION)

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/spawn.hpp"

#endif // ASIO_SPAWN_HPP
