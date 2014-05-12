//
// strand_executor.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_STRAND_EXECUTOR_HPP
#define ASIO_STRAND_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/strand_executor_service.hpp"
#include "asio/is_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// Provides serialised function invocation for any executor type.
template <typename Executor>
class strand_executor
{
public:
  class work;

  /// Default constructor.
  /**
   * This constructor is only valid if the underlying executor type is default
   * constructible.
   */
  strand_executor()
    : executor_(),
      impl_(use_service<detail::strand_executor_service>(
            executor_.context()).create_implementation())
  {
  }

  /// Construct a strand for the specified executor.
  explicit strand_executor(const Executor& e)
    : executor_(e),
      impl_(use_service<detail::strand_executor_service>(
            executor_.context()).create_implementation())
  {
  }

  /// Copy constructor.
  strand_executor(const strand_executor& other)
    : executor_(other.executor_),
      impl_(other.impl_)
  {
  }

  /// Converting constructor.
  /**
   * This constructor is only valid if the @c OtherExecutor type is convertible
   * to @c Executor.
   */
  template <class OtherExecutor>
  strand_executor(const strand_executor<OtherExecutor>& other)
    : executor_(other.executor_),
      impl_(other.impl_)
  {
  }

  /// Assignment operator.
  strand_executor& operator=(const strand_executor& other)
  {
    executor_ = other.executor_;
    impl_ = other.impl_;
    return *this;
  }

  /// Converting assignment operator.
  /**
   * This assignment operator is only valid if the @c OtherExecutor type is
   * convertible to @c Executor.
   */
  template <class OtherExecutor>
  strand_executor& operator=(const strand_executor<OtherExecutor>& other)
  {
    executor_ = other.executor_;
    impl_ = other.impl_;
    return *this;
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move constructor.
  strand_executor(strand_executor&& other)
    : executor_(ASIO_MOVE_CAST(Executor)(other.executor_)),
      impl_(ASIO_MOVE_CAST(implementation_type)(other.impl_))
  {
  }

  /// Converting move constructor.
  /**
   * This constructor is only valid if the @c OtherExecutor type is convertible
   * to @c Executor.
   */
  template <class OtherExecutor>
  strand_executor(strand_executor<OtherExecutor>&& other)
    : executor_(ASIO_MOVE_CAST(OtherExecutor)(other)),
      impl_(ASIO_MOVE_CAST(implementation_type)(other.impl_))
  {
  }

  /// Move assignment operator.
  strand_executor& operator=(strand_executor&& other)
  {
    executor_ = ASIO_MOVE_CAST(Executor)(other);
    impl_ = ASIO_MOVE_CAST(implementation_type)(other.impl_);
    return *this;
  }

  /// Converting move assignment operator.
  /**
   * This assignment operator is only valid if the @c OtherExecutor type is
   * convertible to @c Executor.
   */
  template <class OtherExecutor>
  strand_executor& operator=(const strand_executor<OtherExecutor>&& other)
  {
    executor_ = ASIO_MOVE_CAST(OtherExecutor)(other);
    impl_ = ASIO_MOVE_CAST(implementation_type)(other.impl_);
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destructor.
  ~strand_executor()
  {
  }

  /// Obtain the underlying execution context.
  execution_context& context()
  {
    return executor_.context();
  }

  /// Request the strand to invoke the given function object.
  /**
   * This function is used to ask the strand to execute the given function
   * object on its underlying executor. The function object will be executed
   * inside this function if the strand is not otherwise busy and if the
   * underlying executor's @c dispatch() function is also able to execute the
   * function before returning.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   */
  template <typename Function>
  void dispatch(ASIO_MOVE_ARG(Function) f)
  {
    detail::strand_executor_service::dispatch(impl_,
        executor_, ASIO_MOVE_CAST(Function)(f));
  }

  /// Request the system executor to invoke the given function object.
  /**
   * This function is used to ask the executor to execute the given function
   * object. The function object will never be executed inside this function.
   * Instead, it will be scheduled to run on an unspecified system thread pool.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   */
  template <typename Function>
  void post(ASIO_MOVE_ARG(Function) f)
  {
    detail::strand_executor_service::post(impl_,
        executor_, ASIO_MOVE_CAST(Function)(f));
  }

  /// Request the system executor to invoke the given function object.
  /**
   * This function is used to ask the executor to execute the given function
   * object. The function object will never be executed inside this function.
   * Instead, it will be scheduled to run on an unspecified system thread pool.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   */
  template <typename Function>
  void defer(ASIO_MOVE_ARG(Function) f)
  {
    detail::strand_executor_service::defer(impl_,
        executor_, ASIO_MOVE_CAST(Function)(f));
  }

  /// Determine whether the strand is running in the current thread.
  /**
   * @return @c true if the current thread is executing a function that was
   * submitted to the strand using post(), dispatch() or defer(). Otherwise
   * returns @c false.
   */
  bool running_in_this_thread() const
  {
    return detail::strand_executor_service::running_in_this_thread(impl_);
  }

private:
  Executor executor_;
  typedef detail::strand_executor_service::implementation_type implementation_type;
  implementation_type impl_;
};

/// Class to inform the strand when it has work to do.
/**
 * The work class is used to inform the strand when work starts and finishes.
 * The strand passes this information to its underlying executor.
 *
 * The work class is copy-constructible so that it may be used as a data member
 * in a handler class. It is not assignable.
 */
template <typename Executor>
class strand_executor<Executor>::work
{
public:
  /// Constructor notifies the strand that work is starting.
  explicit work(strand_executor<Executor>& e)
    : impl_(e)
  {
  }

  /// Copy constructor notifies the strand that work is continuing.
  work(const work& other)
    : impl_(other.impl_)
  {
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Copy constructor notifies the strand that work is continuing.
  work(work&& other)
    : impl_(ASIO_MOVE_CAST(typename Executor::work)(other.impl_))
  {
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destructor notifies the strand that the work is complete.
  ~work()
  {
  }

private:
  // Prevent assignment.
  void operator=(const work& other);

  // The underlying executor's work.
  typename Executor::work impl_;
};

#if !defined(GENERATING_DOCUMENTATION)
template <typename Executor>
struct is_executor<strand_executor<Executor> > : true_type {};
#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_STRAND_EXECUTOR_HPP
