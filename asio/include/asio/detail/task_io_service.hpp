//
// task_io_service.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TASK_IO_SERVICE_HPP
#define ASIO_DETAIL_TASK_IO_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/error_code.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/call_stack.hpp"
#include "asio/detail/completion_handler.hpp"
#include "asio/detail/event.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/service_base.hpp"
#include "asio/detail/task_io_service_fwd.hpp"
#include "asio/detail/task_io_service_operation.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/detail/atomic_count.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

template <typename Task>
class task_io_service
  : public asio::detail::service_base<task_io_service<Task> >
{
public:
  typedef task_io_service_operation<Task> operation;

  // Constructor.
  task_io_service(asio::io_service& io_service)
    : asio::detail::service_base<task_io_service<Task> >(io_service),
      mutex_(),
      task_(0),
      task_interrupted_(true),
      outstanding_work_(0),
      stopped_(false),
      shutdown_(false),
      first_idle_thread_(0)
  {
  }

  void init(size_t /*concurrency_hint*/)
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    shutdown_ = true;
    lock.unlock();

    // Destroy handler objects.
    while (!op_queue_.empty())
    {
      operation* o = op_queue_.front();
      op_queue_.pop();
      if (o != &task_operation_)
        o->destroy();
    }

    // Reset to initial state.
    task_ = 0;
  }

  // Initialise the task, if required.
  void init_task()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (!shutdown_ && !task_)
    {
      task_ = &use_service<Task>(this->get_io_service());
      op_queue_.push(&task_operation_);
      wake_one_thread_and_unlock(lock);
    }
  }

  // Run the event loop until interrupted or no more work.
  size_t run(asio::error_code& ec)
  {
    ec = asio::error_code();
    if (outstanding_work_ == 0)
    {
      stop();
      return 0;
    }

    typename call_stack<task_io_service>::context ctx(this);

    idle_thread_info this_idle_thread;
    this_idle_thread.next = 0;

    asio::detail::mutex::scoped_lock lock(mutex_);

    size_t n = 0;
    for (; do_one(lock, &this_idle_thread); lock.lock())
      if (n != (std::numeric_limits<size_t>::max)())
        ++n;
    return n;
  }

  // Run until interrupted or one operation is performed.
  size_t run_one(asio::error_code& ec)
  {
    ec = asio::error_code();
    if (outstanding_work_ == 0)
    {
      stop();
      return 0;
    }

    typename call_stack<task_io_service>::context ctx(this);

    idle_thread_info this_idle_thread;
    this_idle_thread.next = 0;

    asio::detail::mutex::scoped_lock lock(mutex_);

    return do_one(lock, &this_idle_thread);
  }

  // Poll for operations without blocking.
  size_t poll(asio::error_code& ec)
  {
    if (outstanding_work_ == 0)
    {
      stop();
      ec = asio::error_code();
      return 0;
    }

    typename call_stack<task_io_service>::context ctx(this);

    asio::detail::mutex::scoped_lock lock(mutex_);

    size_t n = 0;
    for (; do_one(lock, 0); lock.lock())
      if (n != (std::numeric_limits<size_t>::max)())
        ++n;
    return n;
  }

  // Poll for one operation without blocking.
  size_t poll_one(asio::error_code& ec)
  {
    ec = asio::error_code();
    if (outstanding_work_ == 0)
    {
      stop();
      return 0;
    }

    typename call_stack<task_io_service>::context ctx(this);

    asio::detail::mutex::scoped_lock lock(mutex_);

    return do_one(lock, 0);
  }

  // Interrupt the event processing loop.
  void stop()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    stop_all_threads(lock);
  }

  // Reset in preparation for a subsequent run invocation.
  void reset()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    stopped_ = false;
  }

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
  void dispatch(Handler handler)
  {
    if (call_stack<task_io_service>::contains(this))
    {
      asio::detail::fenced_block b;
      asio_handler_invoke_helpers::invoke(handler, handler);
    }
    else
      post(handler);
  }

  // Request invocation of the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef completion_handler<Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, handler);

    post_immediate_completion(ptr.get());
    ptr.release();
  }

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() has not yet been called for the operation.
  void post_immediate_completion(operation* op)
  {
    work_started();
    post_deferred_completion(op);
  }

  // Request invocation of the given operation and return immediately. Assumes
  // that work_started() was previously called for the operation.
  void post_deferred_completion(operation* op)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    op_queue_.push(op);
    wake_one_thread_and_unlock(lock);
  }

  // Request invocation of the given operations and return immediately. Assumes
  // that work_started() was previously called for each operation.
  void post_deferred_completions(op_queue<operation>& ops)
  {
    if (!ops.empty())
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      op_queue_.push(ops);
      wake_one_thread_and_unlock(lock);
    }
  }

private:
  struct idle_thread_info;

  size_t do_one(asio::detail::mutex::scoped_lock& lock,
      idle_thread_info* this_idle_thread)
  {
    bool polling = !this_idle_thread;
    bool task_has_run = false;
    while (!stopped_)
    {
      if (!op_queue_.empty())
      {
        // Prepare to execute first handler from queue.
        operation* o = op_queue_.front();
        op_queue_.pop();
        bool more_handlers = (!op_queue_.empty());

        if (o == &task_operation_)
        {
          task_interrupted_ = more_handlers || polling;

          // If the task has already run and we're polling then we're done.
          if (task_has_run && polling)
          {
            task_interrupted_ = true;
            op_queue_.push(&task_operation_);
            return 0;
          }
          task_has_run = true;

          if (!more_handlers || !wake_one_idle_thread_and_unlock(lock))
            lock.unlock();

          op_queue<operation> completed_ops;
          task_cleanup c = { this, &lock, &completed_ops };
          (void)c;

          // Run the task. May throw an exception. Only block if the operation
          // queue is empty and we're not polling, otherwise we want to return
          // as soon as possible.
          task_->run(!more_handlers && !polling, completed_ops);
        }
        else
        {
          if (more_handlers)
            wake_one_thread_and_unlock(lock);
          else
            lock.unlock();

          // Ensure the count of outstanding work is decremented on block exit.
          work_finished_on_block_exit on_exit = { this };
          (void)on_exit;

          // Complete the operation. May throw an exception.
          o->complete(*this); // deletes the operation object

          return 1;
        }
      }
      else if (this_idle_thread)
      {
        // Nothing to run right now, so just wait for work to do.
        this_idle_thread->next = first_idle_thread_;
        first_idle_thread_ = this_idle_thread;
        this_idle_thread->wakeup_event.clear(lock);
        this_idle_thread->wakeup_event.wait(lock);
      }
      else
      {
        return 0;
      }
    }

    return 0;
  }

  // Stop the task and all idle threads.
  void stop_all_threads(
      asio::detail::mutex::scoped_lock& lock)
  {
    stopped_ = true;

    while (first_idle_thread_)
    {
      idle_thread_info* idle_thread = first_idle_thread_;
      first_idle_thread_ = idle_thread->next;
      idle_thread->next = 0;
      idle_thread->wakeup_event.signal(lock);
    }

    if (!task_interrupted_ && task_)
    {
      task_interrupted_ = true;
      task_->interrupt();
    }
  }

  // Wakes a single idle thread and unlocks the mutex. Returns true if an idle
  // thread was found. If there is no idle thread, returns false and leaves the
  // mutex locked.
  bool wake_one_idle_thread_and_unlock(
      asio::detail::mutex::scoped_lock& lock)
  {
    if (first_idle_thread_)
    {
      idle_thread_info* idle_thread = first_idle_thread_;
      first_idle_thread_ = idle_thread->next;
      idle_thread->next = 0;
      idle_thread->wakeup_event.signal_and_unlock(lock);
      return true;
    }
    return false;
  }

  // Wake a single idle thread, or the task, and always unlock the mutex.
  void wake_one_thread_and_unlock(
      asio::detail::mutex::scoped_lock& lock)
  {
    if (!wake_one_idle_thread_and_unlock(lock))
    {
      if (!task_interrupted_ && task_)
      {
        task_interrupted_ = true;
        task_->interrupt();
      }
      lock.unlock();
    }
  }

  // Helper class to perform task-related operations on block exit.
  struct task_cleanup;
  friend struct task_cleanup;
  struct task_cleanup
  {
    ~task_cleanup()
    {
      // Enqueue the completed operations and reinsert the task at the end of
      // the operation queue.
      lock_->lock();
      task_io_service_->task_interrupted_ = true;
      task_io_service_->op_queue_.push(*ops_);
      task_io_service_->op_queue_.push(&task_io_service_->task_operation_);
    }

    task_io_service* task_io_service_;
    asio::detail::mutex::scoped_lock* lock_;
    op_queue<operation>* ops_;
  };

  // Helper class to call work_finished() on block exit.
  struct work_finished_on_block_exit
  {
    ~work_finished_on_block_exit()
    {
      task_io_service_->work_finished();
    }

    task_io_service* task_io_service_;
  };

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The task to be run by this service.
  Task* task_;

  // Operation object to represent the position of the task in the queue.
  struct task_operation : public operation
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

  // Structure containing information about an idle thread.
  struct idle_thread_info
  {
    event wakeup_event;
    idle_thread_info* next;
  };

  // The threads that are currently idle.
  idle_thread_info* first_idle_thread_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TASK_IO_SERVICE_HPP
