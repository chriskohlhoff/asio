//
// thread.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_THREAD_HPP
#define ASIO_THREAD_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/thread.hpp"

namespace asio {

/// A simple abstraction for starting threads.
/**
 * The asio::thread class implements the smallest possible subset of the
 * functionality of boost::thread. It is intended to be used only for starting
 * a thread and waiting for it to exit. If more extensive threading
 * capabilities are required, you are strongly advised to use something else.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Example:
 * A typical use of asio::thread would be to launch a thread to run a demuxer's
 * event processing loop:
 *
 * @par
 * @code asio::demuxer d;
 * // ...
 * asio::thread t(boost::bind(&asio::demuxer::run, &d));
 * // ...
 * t.join(); @endcode
 */
class thread
  : private boost::noncopyable
{
public:
  /// Start a new thread that executes the supplied function.
  /**
   * This constructor creates a new thread that will execute the given function
   * or function object.
   *
   * @param f The function or function object to be run in the thread. The
   * equivalent function signature must be: @code void f(); @endcode
   */
  template <typename Function>
  explicit thread(Function f)
    : impl_(f)
  {
  }

  /// Destructor.
  ~thread()
  {
  }

  /// Wait for the thread to exit.
  /**
   * This function will block until the thread has exited.
   *
   * If this function is not called before the thread object is destroyed, the
   * thread itself will continue to run until completion. You will, however,
   * no longer have the ability to wait for it to exit.
   */
  void join()
  {
    impl_.join();
  }

private:
  detail::thread impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_THREAD_HPP
