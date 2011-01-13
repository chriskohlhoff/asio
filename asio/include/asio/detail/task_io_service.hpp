//
// detail/task_io_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TASK_IO_SERVICE_HPP
#define ASIO_DETAIL_TASK_IO_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_IOCP)

#include <boost/detail/atomic_count.hpp>
#include "asio/error_code.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/reactor_fwd.hpp"
#include "asio/detail/task_io_service_fwd.hpp"
#include "asio/detail/task_io_service_operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class task_io_service
  : public asio::detail::service_base<task_io_service>
{
public:
  typedef task_io_service_operation operation;

  // Constructor.
  ASIO_DECL task_io_service(asio::io_service& io_service);

  // How many concurrent threads are likely to run the io_service.
  ASIO_DECL void init(std::size_t concurrency_hint);

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown_service();

  // Initialise the task, if required.
  ASIO_DECL void init_task();

  // Run the event loop until interrupted or no more work.
  ASIO_DECL std::size_t run(asio::error_code& ec);

  // Run until interrupted or one operation is performed.
  ASIO_DECL std::size_t run_one(asio::error_code& ec);

  // Poll for operations without blocking.
  ASIO_DECL std::size_t poll(asio::error_code& ec);

  // Poll for one operation without blocking.
  ASIO_DECL std::size_t poll_one(asio::error_code& ec);

  // Interrupt the event processing loop.
  ASIO_DECL void stop();

  // Reset in preparation for a subsequent run invocation.
  ASIO_DECL void reset();

  // Notify that some work has started.
  void work_started()
  {
    ++outstanding_work_;
  }

  // Notify that some work has finished.
  void work_finished()
  {
    if (--outstanding_work_ == 0)
      stop();
  }

  // Request invocation of the given handler.
  template <typename Handler>
  void dispatch(Handler handler);

  // Request invocation of the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler);

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() has not yet been called for the operation.
  ASIO_DECL void post_immediate_completion(operation* op);

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() was previously called for the operation.
  ASIO_DECL void post_deferred_completion(operation* op);

  // Request invocation of the given operations and return immediately. Assumes
  // that work_started() was previously called for each operation.
  ASIO_DECL void post_deferred_completions(op_queue<operation>& ops);

private:
  // Structure containing information about an idle thread.
  struct idle_thread_info;

  // Run at most one operation. Blocks only if this_idle_thread is non-null.
  ASIO_DECL std::size_t do_one(mutex::scoped_lock& lock,
      idle_thread_info* this_idle_thread);

  // Stop the task and all idle threads.
  ASIO_DECL void stop_all_threads(mutex::scoped_lock& lock);

  // Wakes a single idle thread and unlocks the mutex. Returns true if an idle
  // thread was found. If there is no idle thread, returns false and leaves the
  // mutex locked.
  ASIO_DECL bool wake_one_idle_thread_and_unlock(
      mutex::scoped_lock& lock);

  // Wake a single idle thread, or the task, and always unlock the mutex.
  ASIO_DECL void wake_one_thread_and_unlock(
      mutex::scoped_lock& lock);

  // Helper class to perform task-related operations on block exit.
  struct task_cleanup;
  friend struct task_cleanup;

  // Helper class to call work_finished() on block exit.
  struct work_finished_on_block_exit;

  // Mutex to protect access to internal data.
  mutex mutex_;

  // The task to be run by this service.
  reactor* task_;

  // Operation object to represent the position of the task in the queue.
  struct task_operation : operation
  {
    task_operation() : operation(0) {}
  } task_operation_;

  // Whether the task has been interrupted.
  bool task_interrupted_;

  // The count of unfinished work.
  boost::detail::atomic_count outstanding_work_;

  // The queue of handlers that are ready to be delivered.
  op_queue<operation> op_queue_;

  // Flag to indicate that the dispatcher has been stopped.
  bool stopped_;

  // Flag to indicate that the dispatcher has been shut down.
  bool shutdown_;

  // The threads that are currently idle.
  idle_thread_info* first_idle_thread_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#include "asio/detail/impl/task_io_service.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/task_io_service.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // !defined(ASIO_HAS_IOCP)

#endif // ASIO_DETAIL_TASK_IO_SERVICE_HPP
