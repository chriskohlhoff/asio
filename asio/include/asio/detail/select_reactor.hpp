//
// select_reactor.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SELECT_REACTOR_HPP
#define ASIO_DETAIL_SELECT_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/socket_types.hpp" // Must come before posix_time.

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/io_service.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/fd_set_adapter.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/reactor_op_queue.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/select_reactor_fwd.hpp"
#include "asio/detail/service_base.hpp"
#include "asio/detail/signal_blocker.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/thread.hpp"
#include "asio/detail/timer_op.hpp"
#include "asio/detail/timer_queue_base.hpp"
#include "asio/detail/timer_queue_fwd.hpp"
#include "asio/detail/timer_queue_set.hpp"

namespace asio {
namespace detail {

template <bool Own_Thread>
class select_reactor
  : public asio::detail::service_base<select_reactor<Own_Thread> >
{
public:
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
  enum { read_op = 0, write_op = 1, except_op = 2,
    max_select_ops = 3, connect_op = 3, max_ops = 4 };
#else // defined(BOOST_WINDOWS) || defined(__CYGWIN__)
  enum { read_op = 0, write_op = 1, except_op = 2,
    max_select_ops = 3, connect_op = 1, max_ops = 3 };
#endif // defined(BOOST_WINDOWS) || defined(__CYGWIN__)

  // Per-descriptor data.
  struct per_descriptor_data
  {
  };

  // Constructor.
  select_reactor(asio::io_service& io_service)
    : asio::detail::service_base<
        select_reactor<Own_Thread> >(io_service),
      io_service_(use_service<io_service_impl>(io_service)),
      mutex_(),
      interrupter_(),
      stop_thread_(false),
      thread_(0),
      shutdown_(false)
  {
    if (Own_Thread)
    {
      asio::detail::signal_blocker sb;
      thread_ = new asio::detail::thread(
          bind_handler(&select_reactor::call_run_thread, this));
    }
  }

  // Destructor.
  ~select_reactor()
  {
    shutdown_service();
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    shutdown_ = true;
    stop_thread_ = true;
    lock.unlock();

    if (Own_Thread)
    {
      if (thread_)
      {
        interrupter_.interrupt();
        thread_->join();
        delete thread_;
        thread_ = 0;
      }
    }

    op_queue<operation> ops;

    for (int i = 0; i < max_ops; ++i)
      op_queue_[i].get_all_operations(ops);

    timer_queues_.get_all_timers(ops);
  }

  // Initialise the task, but only if the reactor is not in its own thread.
  void init_task()
  {
    io_service_.init_task();
  }

  // Register a socket with the reactor. Returns 0 on success, system error
  // code on failure.
  int register_descriptor(socket_type, per_descriptor_data&)
  {
    return 0;
  }

  // Start a new operation. The reactor operation will be performed when the
  // given descriptor is flagged as ready, or an error has occurred.
  void start_op(int op_type, socket_type descriptor,
      per_descriptor_data&, reactor_op* op, bool)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (!shutdown_)
    {
      bool first = op_queue_[op_type].enqueue_operation(descriptor, op);
      io_service_.work_started();
      if (first)
        interrupter_.interrupt();
    }
  }

  // Cancel all operations associated with the given descriptor. The
  // handlers associated with the descriptor will be invoked with the
  // operation_aborted error.
  void cancel_ops(socket_type descriptor, per_descriptor_data&)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    cancel_ops_unlocked(descriptor, asio::error::operation_aborted);
  }

  // Cancel any operations that are running against the descriptor and remove
  // its registration from the reactor.
  void close_descriptor(socket_type descriptor, per_descriptor_data&)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    cancel_ops_unlocked(descriptor, asio::error::operation_aborted);
  }

  // Add a new timer queue to the reactor.
  template <typename Time_Traits>
  void add_timer_queue(timer_queue<Time_Traits>& timer_queue)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    timer_queues_.insert(&timer_queue);
  }

  // Remove a timer queue from the reactor.
  template <typename Time_Traits>
  void remove_timer_queue(timer_queue<Time_Traits>& timer_queue)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    timer_queues_.erase(&timer_queue);
  }

  // Schedule a new operation in the given timer queue to expire at the
  // specified absolute time.
  template <typename Time_Traits>
  void schedule_timer(timer_queue<Time_Traits>& timer_queue,
      const typename Time_Traits::time_type& time, timer_op* op, void* token)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (!shutdown_)
    {
      bool earliest = timer_queue.enqueue_timer(time, op, token);
      io_service_.work_started();
      if (earliest)
        interrupter_.interrupt();
    }
  }

  // Cancel the timer operations associated with the given token. Returns the
  // number of operations that have been posted or dispatched.
  template <typename Time_Traits>
  std::size_t cancel_timer(timer_queue<Time_Traits>& timer_queue, void* token)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    op_queue<operation> ops;
    std::size_t n = timer_queue.cancel_timer(token, ops);
    lock.unlock();
    io_service_.post_deferred_completions(ops);
    return n;
  }

  // Run select once until interrupted or events are ready to be dispatched.
  void run(bool block, op_queue<operation>& ops)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // Check if the thread is supposed to stop.
    if (Own_Thread)
      if (stop_thread_)
        return;

    // Set up the descriptor sets.
    fd_set_adapter fds[max_select_ops];
    fds[read_op].set(interrupter_.read_descriptor());
    socket_type max_fd = 0;
    bool have_work_to_do = !timer_queues_.all_empty();
    for (int i = 0; i < max_select_ops; ++i)
    {
      have_work_to_do = have_work_to_do || !op_queue_[i].empty();
      op_queue_[i].get_descriptors(fds[i], ops);
      if (fds[i].max_descriptor() > max_fd)
        max_fd = fds[i].max_descriptor();
    }

#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
    // Connection operations on Windows use both except and write fd_sets.
    have_work_to_do = have_work_to_do || !op_queue_[connect_op].empty();
    op_queue_[connect_op].get_descriptors(fds[write_op], ops);
    if (fds[write_op].max_descriptor() > max_fd)
      max_fd = fds[write_op].max_descriptor();
    op_queue_[connect_op].get_descriptors(fds[except_op], ops);
    if (fds[except_op].max_descriptor() > max_fd)
      max_fd = fds[except_op].max_descriptor();
#endif // defined(BOOST_WINDOWS) || defined(__CYGWIN__)

    // We can return immediately if there's no work to do and the reactor is
    // not supposed to block.
    if (!block && !have_work_to_do)
      return;

    // Determine how long to block while waiting for events.
    timeval tv_buf = { 0, 0 };
    timeval* tv = block ? get_timeout(tv_buf) : &tv_buf;

    lock.unlock();

    // Block on the select call until descriptors become ready.
    asio::error_code ec;
    int retval = socket_ops::select(static_cast<int>(max_fd + 1),
        fds[read_op], fds[write_op], fds[except_op], tv, ec);

    // Reset the interrupter.
    if (retval > 0 && fds[read_op].is_set(interrupter_.read_descriptor()))
      interrupter_.reset();

    lock.lock();

    // Dispatch all ready operations.
    if (retval > 0)
    {
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
      // Connection operations on Windows use both except and write fd_sets.
      op_queue_[connect_op].perform_operations_for_descriptors(
          fds[except_op], ops);
      op_queue_[connect_op].perform_operations_for_descriptors(
          fds[write_op], ops);
#endif // defined(BOOST_WINDOWS) || defined(__CYGWIN__)

      // Exception operations must be processed first to ensure that any
      // out-of-band data is read before normal data.
      for (int i = max_select_ops - 1; i >= 0; --i)
        op_queue_[i].perform_operations_for_descriptors(fds[i], ops);
    }
    timer_queues_.get_ready_timers(ops);
  }

  // Interrupt the select loop.
  void interrupt()
  {
    interrupter_.interrupt();
  }

private:
  // Run the select loop in the thread.
  void run_thread()
  {
    if (Own_Thread)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      while (!stop_thread_)
      {
        lock.unlock();
        op_queue<operation> ops;
        run(true, ops);
        io_service_.post_deferred_completions(ops);
        lock.lock();
      }
    }
  }

  // Entry point for the select loop thread.
  static void call_run_thread(select_reactor* reactor)
  {
    if (Own_Thread)
    {
      reactor->run_thread();
    }
  }

  // Get the timeout value for the select call.
  timeval* get_timeout(timeval& tv)
  {
    // By default we will wait no longer than 5 minutes. This will ensure that
    // any changes to the system clock are detected after no longer than this.
    long usec = timer_queues_.wait_duration_usec(5 * 60 * 1000 * 1000);
    tv.tv_sec = usec / 1000000;
    tv.tv_usec = usec % 1000000;
    return &tv;
  }

  // Cancel all operations associated with the given descriptor. This function
  // does not acquire the select_reactor's mutex.
  void cancel_ops_unlocked(socket_type descriptor,
      const asio::error_code& ec)
  {
    bool need_interrupt = false;
    op_queue<operation> ops;
    for (int i = 0; i < max_ops; ++i)
      need_interrupt = op_queue_[i].cancel_operations(
          descriptor, ops, ec) || need_interrupt;
    io_service_.post_deferred_completions(ops);
    if (need_interrupt)
      interrupter_.interrupt();
  }

  // The io_service implementation used to post completions.
  io_service_impl& io_service_;

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The interrupter is used to break a blocking select call.
  select_interrupter interrupter_;

  // The queues of read, write and except operations.
  reactor_op_queue<socket_type> op_queue_[max_ops];

  // The timer queues.
  timer_queue_set timer_queues_;

  // Does the reactor loop thread need to stop.
  bool stop_thread_;

  // The thread that is running the reactor loop.
  asio::detail::thread* thread_;

  // Whether the service has been shut down.
  bool shutdown_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SELECT_REACTOR_HPP
