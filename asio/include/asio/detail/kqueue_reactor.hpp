//
// kqueue_reactor.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2005 Stefan Arentz (stefan at soze dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_KQUEUE_REACTOR_HPP
#define ASIO_DETAIL_KQUEUE_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/kqueue_reactor_fwd.hpp"

#if defined(ASIO_HAS_KQUEUE)

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <sys/types.h>
#include <sys/event.h>
#include <sys/time.h>
#include <boost/config.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/reactor_op_queue.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/service_base.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/timer_op.hpp"
#include "asio/detail/timer_queue_base.hpp"
#include "asio/detail/timer_queue_fwd.hpp"
#include "asio/detail/timer_queue_set.hpp"

// Older versions of Mac OS X may not define EV_OOBAND.
#if !defined(EV_OOBAND)
# define EV_OOBAND EV_FLAG1
#endif // !defined(EV_OOBAND)

namespace asio {
namespace detail {

class kqueue_reactor
  : public asio::detail::service_base<kqueue_reactor>
{
public:
  enum { read_op = 0, write_op = 1,
    connect_op = 1, except_op = 2, max_ops = 3 };

  // Per-descriptor data.
  struct per_descriptor_data
  {
    bool allow_speculative[max_ops];
  };

  // Constructor.
  kqueue_reactor(asio::io_service& io_service)
    : asio::detail::service_base<kqueue_reactor>(io_service),
      io_service_(use_service<io_service_impl>(io_service)),
      mutex_(),
      kqueue_fd_(do_kqueue_create()),
      interrupter_(),
      shutdown_(false),
      need_kqueue_wait_(true)
  {
    // Add the interrupter's descriptor to the kqueue.
    struct kevent event;
    EV_SET(&event, interrupter_.read_descriptor(),
        EVFILT_READ, EV_ADD, 0, 0, 0);
    ::kevent(kqueue_fd_, &event, 1, 0, 0, 0);
  }

  // Destructor.
  ~kqueue_reactor()
  {
    shutdown_service();
    close(kqueue_fd_);
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    shutdown_ = true;
    lock.unlock();

    op_queue<operation> ops;

    for (int i = 0; i < max_ops; ++i)
      op_queue_[i].get_all_operations(ops);

    timer_queues_.get_all_timers(ops);
  }

  // Initialise the task.
  void init_task()
  {
    io_service_.init_task();
  }

  // Register a socket with the reactor. Returns 0 on success, system error
  // code on failure.
  int register_descriptor(socket_type, per_descriptor_data& descriptor_data)
  {
    descriptor_data.allow_speculative[read_op] = true;
    descriptor_data.allow_speculative[write_op] = true;
    descriptor_data.allow_speculative[except_op] = true;

    return 0;
  }

  // Start a new operation. The reactor operation will be performed when the
  // given descriptor is flagged as ready, or an error has occurred.
  void start_op(int op_type, socket_type descriptor,
      per_descriptor_data& descriptor_data,
      reactor_op* op, bool allow_speculative)
  {
    if (allow_speculative && descriptor_data.allow_speculative[op_type])
    {
      if (op->perform())
      {
        io_service_.post_immediate_completion(op);
        return;
      }

      // We only get one shot at a speculative read in this function.
      allow_speculative = false;
    }

    asio::detail::mutex::scoped_lock lock(mutex_);

    if (shutdown_)
      return;

    if (!allow_speculative)
      need_kqueue_wait_ = true;
    else if (!op_queue_[op_type].has_operation(descriptor))
    {
      // Speculative reads are ok as there are no queued read operations.
      descriptor_data.allow_speculative[op_type] = true;

      if (op->perform())
      {
        lock.unlock();
        io_service_.post_immediate_completion(op);
        return;
      }
    }

    // Speculative reads are not ok as there will be queued read operations.
    descriptor_data.allow_speculative[op_type] = false;

    bool first = op_queue_[op_type].enqueue_operation(descriptor, op);
    io_service_.work_started();
    if (first)
    {
      struct kevent event;
      switch (op_type)
      {
      case read_op:
        EV_SET(&event, descriptor, EVFILT_READ, EV_ADD, 0, 0, 0);
        break;
      case write_op:
        EV_SET(&event, descriptor, EVFILT_WRITE, EV_ADD, 0, 0, 0);
        break;
      case except_op:
        if (op_queue_[read_op].has_operation(descriptor))
          EV_SET(&event, descriptor, EVFILT_READ, EV_ADD, 0, 0, 0);
        else
          EV_SET(&event, descriptor, EVFILT_WRITE, EV_ADD, EV_OOBAND, 0, 0);
        break;
      }
      if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
      {
        asio::error_code ec(errno,
            asio::error::get_system_category());
        cancel_ops_unlocked(descriptor, ec);
      }
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

    // Remove the descriptor from kqueue.
    struct kevent event[2];
    EV_SET(&event[0], descriptor, EVFILT_READ, EV_DELETE, 0, 0, 0);
    EV_SET(&event[1], descriptor, EVFILT_WRITE, EV_DELETE, 0, 0, 0);
    ::kevent(kqueue_fd_, event, 2, 0, 0, 0);
    
    // Cancel any outstanding operations associated with the descriptor.
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

  // Run the kqueue loop.
  void run(bool block, op_queue<operation>& ops)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // We can return immediately if there's no work to do and the reactor is
    // not supposed to block.
    if (!block && op_queue_[read_op].empty() && op_queue_[write_op].empty()
        && op_queue_[except_op].empty() && timer_queues_.all_empty())
      return;

    // Determine how long to block while waiting for events.
    timespec timeout_buf = { 0, 0 };
    timespec* timeout = block ? get_timeout(timeout_buf) : &timeout_buf;

    lock.unlock();

    // Block on the kqueue descriptor.
    struct kevent events[128];
    int num_events = (block || need_kqueue_wait_)
      ? kevent(kqueue_fd_, 0, 0, events, 128, timeout)
      : 0;

    lock.lock();

    // Dispatch the waiting events.
    for (int i = 0; i < num_events; ++i)
    {
      int descriptor = events[i].ident;
      if (descriptor == interrupter_.read_descriptor())
      {
        interrupter_.reset();
      }
      else if (events[i].filter == EVFILT_READ)
      {
        // Dispatch operations associated with the descriptor.
        bool more_reads = false;
        bool more_except = false;
        if (events[i].flags & EV_ERROR)
        {
          asio::error_code error(
              events[i].data, asio::error::get_system_category());
          op_queue_[except_op].perform_operations(descriptor, ops);
          op_queue_[read_op].perform_operations(descriptor, ops);
        }
        else if (events[i].flags & EV_OOBAND)
        {
          more_except
            = op_queue_[except_op].perform_operations(descriptor, ops);
          if (events[i].data > 0)
            more_reads = op_queue_[read_op].perform_operations(descriptor, ops);
          else
            more_reads = op_queue_[read_op].has_operation(descriptor);
        }
        else
        {
          more_reads = op_queue_[read_op].perform_operations(descriptor, ops);
          more_except = op_queue_[except_op].has_operation(descriptor);
        }

        // Update the descriptor in the kqueue.
        struct kevent event;
        if (more_reads)
          EV_SET(&event, descriptor, EVFILT_READ, EV_ADD, 0, 0, 0);
        else if (more_except)
          EV_SET(&event, descriptor, EVFILT_READ, EV_ADD, EV_OOBAND, 0, 0);
        else
          EV_SET(&event, descriptor, EVFILT_READ, EV_DELETE, 0, 0, 0);
        if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
        {
          asio::error_code error(errno,
              asio::error::get_system_category());
          op_queue_[except_op].cancel_operations(descriptor, ops, error);
          op_queue_[read_op].cancel_operations(descriptor, ops, error);
        }
      }
      else if (events[i].filter == EVFILT_WRITE)
      {
        // Dispatch operations associated with the descriptor.
        bool more_writes = false;
        if (events[i].flags & EV_ERROR)
        {
          asio::error_code error(
              events[i].data, asio::error::get_system_category());
          op_queue_[write_op].cancel_operations(descriptor, ops, error);
        }
        else
        {
          more_writes = op_queue_[write_op].perform_operations(descriptor, ops);
        }

        // Update the descriptor in the kqueue.
        struct kevent event;
        if (more_writes)
          EV_SET(&event, descriptor, EVFILT_WRITE, EV_ADD, 0, 0, 0);
        else
          EV_SET(&event, descriptor, EVFILT_WRITE, EV_DELETE, 0, 0, 0);
        if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
        {
          asio::error_code error(errno,
              asio::error::get_system_category());
          op_queue_[write_op].cancel_operations(descriptor, ops, error);
        }
      }
    }
    timer_queues_.get_ready_timers(ops);

    // Determine whether kqueue needs to be called next time the reactor is run.
    need_kqueue_wait_ = !op_queue_[read_op].empty()
      || !op_queue_[write_op].empty() || !op_queue_[except_op].empty();
  }

  // Interrupt the select loop.
  void interrupt()
  {
    interrupter_.interrupt();
  }

private:
  // Create the kqueue file descriptor. Throws an exception if the descriptor
  // cannot be created.
  static int do_kqueue_create()
  {
    int fd = kqueue();
    if (fd == -1)
    {
      boost::throw_exception(
          asio::system_error(
            asio::error_code(errno,
              asio::error::get_system_category()),
            "kqueue"));
    }
    return fd;
  }

  // Get the timeout value for the kevent call.
  timespec* get_timeout(timespec& ts)
  {
    // By default we will wait no longer than 5 minutes. This will ensure that
    // any changes to the system clock are detected after no longer than this.
    long usec = timer_queues_.wait_duration_usec(5 * 60 * 1000 * 1000);
    ts.tv_sec = usec / 1000000;
    ts.tv_nsec = (usec % 1000000) * 1000;
    return &ts;
  }

  // Cancel all operations associated with the given descriptor. This function
  // does not acquire the kqueue_reactor's mutex.
  void cancel_ops_unlocked(socket_type descriptor,
      const asio::error_code& ec)
  {
    op_queue<operation> ops;
    for (int i = 0; i < max_ops; ++i)
      op_queue_[i].cancel_operations(descriptor, ops, ec);
    io_service_.post_deferred_completions(ops);
  }

  // The io_service implementation used to post completions.
  io_service_impl& io_service_;

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The kqueue file descriptor.
  int kqueue_fd_;

  // The interrupter is used to break a blocking kevent call.
  select_interrupter interrupter_;

  // The queues of read, write and except operations.
  reactor_op_queue<socket_type> op_queue_[max_ops];

  // The timer queues.
  timer_queue_set timer_queues_;

  // Whether the service has been shut down.
  bool shutdown_;

  // Whether we need to call kqueue the next time the reactor is run.
  bool need_kqueue_wait_;
};

} // namespace detail
} // namespace asio

#endif // defined(ASIO_HAS_KQUEUE)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_KQUEUE_REACTOR_HPP
