//
// system_executor.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SYSTEM_EXECUTOR_HPP
#define ASIO_SYSTEM_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/scheduler.hpp"
#include "asio/detail/thread_group.hpp"
#include "asio/execution_context.hpp"
#include "asio/executor_wrapper.hpp"
#include "asio/is_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// An executor that uses arbitrary threads.
/**
 * The system executor represents an execution context where functions are
 * permitted to run on arbitrary threads. The post() and defer() functions
 * schedule the function to run on an unspecified system thread pool, and
 * dispatch() invokes the function immediately.
 */
class system_executor
{
public:
  /// Tracks outstanding work associated with the executor.
  class work
  {
  public:
    /// Constructor.
    /**
     * For the unspecified executor, this is a no-op.
     */
    explicit work(system_executor)
    {
    }

    /// Destructor.
    /**
     * For the unspecified executor, this is a no-op.
     */
    ~work()
    {
    }
  };

  /// Obtain the underlying execution context.
  execution_context& context();

  /// Request the system executor to invoke the given function object.
  /**
   * This function is used to ask the executor to execute the given function
   * object. The function object will always be executed inside this function.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   */
  template <typename Function>
  void dispatch(ASIO_MOVE_ARG(Function) f);

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
  void post(ASIO_MOVE_ARG(Function) f);

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
  void defer(ASIO_MOVE_ARG(Function) f);

  /// Associate this executor with the specified object.
  template <typename T>
  typename wrap_with_executor_type<T, system_executor>::type wrap(
      ASIO_MOVE_ARG(T) t) const
  {
    return (wrap_with_executor)(ASIO_MOVE_CAST(T)(t), *this);
  }

private:
  struct thread_function;

  // Hidden implementation of the system execution context.
  struct context_impl
    : public execution_context
  {
    // Constructor creates all threads in the system thread pool.
    ASIO_DECL context_impl();

    // Destructor shuts down all threads in the system thread pool.
    ASIO_DECL ~context_impl();

    // The underlying scheduler.
    detail::scheduler& scheduler_;

    // The threads in the system thread pool.
    detail::thread_group threads_;
  };
};

#if !defined(GENERATING_DOCUMENTATION)
template <> struct is_executor<system_executor> : true_type {};
#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/impl/system_executor.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/impl/system_executor.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_SYSTEM_EXECUTOR_HPP
