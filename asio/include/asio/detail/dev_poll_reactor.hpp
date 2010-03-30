//
// dev_poll_reactor.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_DEV_POLL_REACTOR_HPP
#define ASIO_DETAIL_DEV_POLL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/dev_poll_reactor_fwd.hpp"

#if defined(ASIO_HAS_DEV_POLL)

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <vector>
#include <boost/config.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/throw_exception.hpp>
#include <sys/devpoll.h>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/hash_map.hpp"
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

namespace asio {
namespace detail {

class dev_poll_reactor
  : public asio::detail::service_base<dev_poll_reactor>
{
public:
  enum { read_op = 0, write_op = 1,
    connect_op = 1, except_op = 2, max_ops = 3 };

  // Per-descriptor data.
  struct per_descriptor_data
  {
  };

  // Constructor.
  dev_poll_reactor(asio::io_service& io_service)
    : asio::detail::service_base<dev_poll_reactor>(io_service),
      io_service_(use_service<io_service_impl>(io_service)),
      mutex_(),
      dev_poll_fd_(do_dev_poll_create()),
      interrupter_(),
      shutdown_(false)
  {
    // Add the interrupter's descriptor to /dev/poll.
    ::pollfd ev = { 0 };
    ev.fd = interrupter_.read_descriptor();
    ev.events = POLLIN | POLLERR;
    ev.revents = 0;
    ::write(dev_poll_fd_, &ev, sizeof(ev));
  }

  // Destructor.
  ~dev_poll_reactor()
  {
    shutdown_service();
    ::close(dev_poll_fd_);
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
  int register_descriptor(socket_type, per_descriptor_data&)
  {
    return 0;
  }

  // Start a new operation. The reactor operation will be performed when the
  // given descriptor is flagged as ready, or an error has occurred.
  void start_op(int op_type, socket_type descriptor,
      per_descriptor_data&, reactor_op* op, bool allow_speculative)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    if (shutdown_)
      return;

    if (allow_speculative)
    {
      if (op_type != read_op || !op_queue_[except_op].has_operation(descriptor))
      {
        if (!op_queue_[op_type].has_operation(descriptor))
        {
          if (op->perform())
          {
            lock.unlock();
            io_service_.post_immediate_completion(op);
            return;
          }
        }
      }
    }

    bool first = op_queue_[op_type].enqueue_operation(descriptor, op);
    io_service_.work_started();
    if (first)
    {
      ::pollfd& ev = add_pending_event_change(descriptor);
      ev.events = POLLERR | POLLHUP;
      if (op_type == read_op
          || op_queue_[read_op].has_operation(descriptor))
        ev.events |= POLLIN;
      if (op_type == write_op
          || op_queue_[write_op].has_operation(descriptor))
        ev.events |= POLLOUT;
      if (op_type == except_op
          || op_queue_[except_op].has_operation(descriptor))
        ev.events |= POLLPRI;
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

    // Remove the descriptor from /dev/poll.
    ::pollfd& ev = add_pending_event_change(descriptor);
    ev.events = POLLREMOVE;
    interrupter_.interrupt();

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

  // Run /dev/poll once until interrupted or events are ready to be dispatched.
  void run(bool block, op_queue<operation>& ops)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // We can return immediately if there's no work to do and the reactor is
    // not supposed to block.
    if (!block && op_queue_[read_op].empty() && op_queue_[write_op].empty()
        && op_queue_[except_op].empty() && timer_queues_.all_empty())
      return;

    // Write the pending event registration changes to the /dev/poll descriptor.
    std::size_t events_size = sizeof(::pollfd) * pending_event_changes_.size();
    if (events_size > 0)
    {
      errno = 0;
      int result = ::write(dev_poll_fd_,
          &pending_event_changes_[0], events_size);
      if (result != static_cast<int>(events_size))
      {
        asio::error_code ec = asio::error_code(
            errno, asio::error::get_system_category());
        for (std::size_t i = 0; i < pending_event_changes_.size(); ++i)
        {
          int descriptor = pending_event_changes_[i].fd;
          for (int j = 0; j < max_ops; ++j)
            op_queue_[j].cancel_operations(descriptor, ops, ec);
        }
      }
      pending_event_changes_.clear();
      pending_event_change_index_.clear();
    }

    int timeout = block ? get_timeout() : 0;
    lock.unlock();

    // Block on the /dev/poll descriptor.
    ::pollfd events[128] = { { 0 } };
    ::dvpoll dp = { 0 };
    dp.dp_fds = events;
    dp.dp_nfds = 128;
    dp.dp_timeout = timeout;
    int num_events = ::ioctl(dev_poll_fd_, DP_POLL, &dp);

    lock.lock();

    // Dispatch the waiting events.
    for (int i = 0; i < num_events; ++i)
    {
      int descriptor = events[i].fd;
      if (descriptor == interrupter_.read_descriptor())
      {
        interrupter_.reset();
      }
      else
      {
        bool more_reads = false;
        bool more_writes = false;
        bool more_except = false;

        // Exception operations must be processed first to ensure that any
        // out-of-band data is read before normal data.
        if (events[i].events & (POLLPRI | POLLERR | POLLHUP))
          more_except =
            op_queue_[except_op].perform_operations(descriptor, ops);
        else
          more_except = op_queue_[except_op].has_operation(descriptor);

        if (events[i].events & (POLLIN | POLLERR | POLLHUP))
          more_reads = op_queue_[read_op].perform_operations(descriptor, ops);
        else
          more_reads = op_queue_[read_op].has_operation(descriptor);

        if (events[i].events & (POLLOUT | POLLERR | POLLHUP))
          more_writes = op_queue_[write_op].perform_operations(descriptor, ops);
        else
          more_writes = op_queue_[write_op].has_operation(descriptor);

        if ((events[i].events & (POLLERR | POLLHUP)) != 0
              && !more_except && !more_reads && !more_writes)
        {
          // If we have an event and no operations associated with the
          // descriptor then we need to delete the descriptor from /dev/poll.
          // The poll operation can produce POLLHUP or POLLERR events when there
          // is no operation pending, so if we do not remove the descriptor we
          // can end up in a tight polling loop.
          ::pollfd ev = { 0 };
          ev.fd = descriptor;
          ev.events = POLLREMOVE;
          ev.revents = 0;
          ::write(dev_poll_fd_, &ev, sizeof(ev));
        }
        else
        {
          ::pollfd ev = { 0 };
          ev.fd = descriptor;
          ev.events = POLLERR | POLLHUP;
          if (more_reads)
            ev.events |= POLLIN;
          if (more_writes)
            ev.events |= POLLOUT;
          if (more_except)
            ev.events |= POLLPRI;
          ev.revents = 0;
          int result = ::write(dev_poll_fd_, &ev, sizeof(ev));
          if (result != sizeof(ev))
          {
            asio::error_code ec(errno,
                asio::error::get_system_category());
            for (int j = 0; j < max_ops; ++j)
              op_queue_[j].cancel_operations(descriptor, ops, ec);
          }
        }
      }
    }
    timer_queues_.get_ready_timers(ops);
  }

  // Interrupt the select loop.
  void interrupt()
  {
    interrupter_.interrupt();
  }

private:
  // Create the /dev/poll file descriptor. Throws an exception if the descriptor
  // cannot be created.
  static int do_dev_poll_create()
  {
    int fd = ::open("/dev/poll", O_RDWR);
    if (fd == -1)
    {
      boost::throw_exception(
          asio::system_error(
            asio::error_code(errno,
              asio::error::get_system_category()),
            "/dev/poll"));
    }
    return fd;
  }

  // Get the timeout value for the /dev/poll DP_POLL operation. The timeout
  // value is returned as a number of milliseconds. A return value of -1
  // indicates that the poll should block indefinitely.
  int get_timeout()
  {
    // By default we will wait no longer than 5 minutes. This will ensure that
    // any changes to the system clock are detected after no longer than this.
    return timer_queues_.wait_duration_msec(5 * 60 * 1000);
  }

  // Cancel all operations associated with the given descriptor. The do_cancel
  // function of the handler objects will be invoked. This function does not
  // acquire the dev_poll_reactor's mutex.
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

  // Add a pending event entry for the given descriptor.
  ::pollfd& add_pending_event_change(int descriptor)
  {
    hash_map<int, std::size_t>::iterator iter
      = pending_event_change_index_.find(descriptor);
    if (iter == pending_event_change_index_.end())
    {
      std::size_t index = pending_event_changes_.size();
      pending_event_changes_.reserve(pending_event_changes_.size() + 1);
      pending_event_change_index_.insert(std::make_pair(descriptor, index));
      pending_event_changes_.push_back(::pollfd());
      pending_event_changes_[index].fd = descriptor;
      pending_event_changes_[index].revents = 0;
      return pending_event_changes_[index];
    }
    else
    {
      return pending_event_changes_[iter->second];
    }
  }

  // The io_service implementation used to post completions.
  io_service_impl& io_service_;

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The /dev/poll file descriptor.
  int dev_poll_fd_;

  // Vector of /dev/poll events waiting to be written to the descriptor.
  std::vector< ::pollfd> pending_event_changes_;

  // Hash map to associate a descriptor with a pending event change index.
  hash_map<int, std::size_t> pending_event_change_index_;

  // The interrupter is used to break a blocking DP_POLL operation.
  select_interrupter interrupter_;

  // The queues of read, write and except operations.
  reactor_op_queue<socket_type> op_queue_[max_ops];

  // The timer queues.
  timer_queue_set timer_queues_;

  // Whether the service has been shut down.
  bool shutdown_;
};

} // namespace detail
} // namespace asio

#endif // defined(ASIO_HAS_DEV_POLL)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_DEV_POLL_REACTOR_HPP
