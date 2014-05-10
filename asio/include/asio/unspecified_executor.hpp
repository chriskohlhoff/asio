//
// unspecified_executor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_UNSPECIFIED_EXECUTOR_HPP
#define ASIO_UNSPECIFIED_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution_context.hpp"
#include "asio/is_executor.hpp"
#include "asio/system_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// An executor for functions that do not explicitly specify one.
/**
 * The unspecified executor is used for function objects that do not explicitly
 * specify an associated executor. It is implemented in terms of the system
 * executor.
 */
class unspecified_executor
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
    explicit work(unspecified_executor)
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
  execution_context& context()
  {
    return system_executor().context();
  }

  /// Request the unspecified executor to invoke the given function object.
  /**
   * This function is used to ask the executor to execute the given function
   * object. The function object will always be executed inside this function.
   *
   * @param f The function object to be called. The executor will make
   * a copy of the handler object as required. The function signature of the
   * function object must be: @code void function(); @endcode
   */
  template <typename Function>
  void dispatch(ASIO_MOVE_ARG(Function) f)
  {
    system_executor().dispatch(ASIO_MOVE_CAST(Function)(f));
  }

  /// Request the unspecified executor to invoke the given function object.
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
    system_executor().post(ASIO_MOVE_CAST(Function)(f));
  }

  /// Request the unspecified executor to invoke the given function object.
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
    system_executor().defer(ASIO_MOVE_CAST(Function)(f));
  }
};

#if !defined(GENERATING_DOCUMENTATION)
template <> struct is_executor<unspecified_executor> : true_type {};
#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_UNSPECIFIED_EXECUTOR_HPP
