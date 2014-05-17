//
// executor_work.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTOR_WORK_HPP
#define ASIO_EXECUTOR_WORK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// An object of type @c executor_work controls ownership of executor work within
/// a scope.
template <typename Executor>
class executor_work
{
public:
  /// The underlying executor type.
  typedef Executor executor_type;

  /// Constructs a @c executor_work object for the specified executor.
  /**
   * Stores a copy of @c e and calls <tt>work_started()</tt> on it.
   */
  explicit executor_work(const executor_type& e) ASIO_NOEXCEPT
    : executor_(e),
      owns_(true)
  {
    executor_.work_started();
  }

  /// Copy constructor.
  executor_work(const executor_work& other) ASIO_NOEXCEPT
    : executor_(other.executor_),
      owns_(other.owns_)
  {
    if (owns_)
      executor_.work_started();
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move constructor.
  executor_work(executor_work&& other)
    : executor_(ASIO_MOVE_CAST(Executor)(other.executor_)),
      owns_(other.owns_)
  {
    other.owns_ = false;
  }
#endif //  defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destructor.
  /**
   * Unless the object is in a moved-from state, calls <tt>work_finished()</tt>
   * on the stored executor.
   */
  ~executor_work()
  {
    if (owns_)
      executor_.work_finished();
  }

  /// Obtain the associated executor.
  executor_type get_executor() const ASIO_NOEXCEPT
  {
    return executor_;
  }

private:
  // Disallow assignment.
  executor_work& operator=(const executor_work&);

  executor_type executor_;
  bool owns_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTOR_WORK_HPP
