//
// epoll_reactor.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_EPOLL_REACTOR_HPP
#define ASIO_DETAIL_EPOLL_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/epoll_reactor_fwd.hpp"

#if defined(ASIO_HAS_EPOLL)

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <sys/epoll.h>
#include <boost/config.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/hash_map.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/service_base.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/timer_op.hpp"
#include "asio/detail/timer_queue_base.hpp"
#include "asio/detail/timer_queue_fwd.hpp"
#include "asio/detail/timer_queue_set.hpp"

#if (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 8)
# define ASIO_HAS_TIMERFD 1
#endif // (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 8)

#if defined(ASIO_HAS_TIMERFD)
# include "asio/detail/push_options.hpp"
# include <sys/timerfd.h>
# include "asio/detail/pop_options.hpp"
#endif // defined(ASIO_HAS_TIMERFD)

namespace asio {
namespace detail {

class epoll_reactor
  : public asio::detail::service_base<epoll_reactor>
{
public:
  enum { read_op = 0, write_op = 1,
    connect_op = 1, except_op = 2, max_ops = 3 };

  // Per-descriptor queues.
  struct descriptor_state
  {
    descriptor_state() {}
    descriptor_state(const descriptor_state&) {}
    void operator=(const descriptor_state&) {}

    mutex mutex_;
    op_queue<reactor_op> op_queue_[max_ops];
    bool shutdown_;
  };

  // Per-descriptor data.
  typedef descriptor_state* per_descriptor_data;

  // Constructor.
  epoll_reactor(asio::io_service& io_service)
    : asio::detail::service_base<epoll_reactor>(io_service),
      io_service_(use_service<io_service_impl>(io_service)),
      mutex_(),
      epoll_fd_(do_epoll_create()),
#if defined(ASIO_HAS_TIMERFD)
      timer_fd_(timerfd_create(CLOCK_MONOTONIC, 0)),
#else // defined(ASIO_HAS_TIMERFD)
      timer_fd_(-1),
#endif // defined(ASIO_HAS_TIMERFD)
      interrupter_(),
      shutdown_(false)
  {
    // Add the interrupter's descriptor to epoll.
    epoll_event ev = { 0, { 0 } };
    ev.events = EPOLLIN | EPOLLERR | EPOLLET;
    ev.data.ptr = &interrupter_;
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, interrupter_.read_descriptor(), &ev);
    interrupter_.interrupt();

    // Add the timer descriptor to epoll.
    if (timer_fd_ != -1)
    {
      ev.events = EPOLLIN | EPOLLERR;
      ev.data.ptr = &timer_fd_;
      epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, timer_fd_, &ev);
    }
  }

  // Destructor.
  ~epoll_reactor()
  {
    close(epoll_fd_);
    if (timer_fd_ != -1)
      close(timer_fd_);
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    mutex::scoped_lock lock(mutex_);
    shutdown_ = true;
    lock.unlock();

    op_queue<operation> ops;

    descriptor_map::iterator iter = registered_descriptors_.begin();
    descriptor_map::iterator end = registered_descriptors_.end();
    while (iter != end)
    {
      for (int i = 0; i < max_ops; ++i)
        ops.push(iter->second.op_queue_[i]);
      iter->second.shutdown_ = true;
      ++iter;
    }

    timer_queues_.get_all_timers(ops);
  }

  // Initialise the task.
  void init_task()
  {
    io_service_.init_task();
  }

  // Register a socket with the reactor. Returns 0 on success, system error
  // code on failure.
  int register_descriptor(socket_type descriptor,
      per_descriptor_data& descriptor_data)
  {
    mutex::scoped_lock lock(registered_descriptors_mutex_);

    descriptor_map::iterator new_entry = registered_descriptors_.insert(
          std::make_pair(descriptor, descriptor_state())).first;
    descriptor_data = &new_entry->second;

    epoll_event ev = { 0, { 0 } };
    ev.events = EPOLLIN | EPOLLERR | EPOLLHUP | EPOLLOUT | EPOLLPRI | EPOLLET;
    ev.data.ptr = descriptor_data;
    int result = epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, descriptor, &ev);
    if (result != 0)
      return errno;

    descriptor_data->shutdown_ = false;

    return 0;
  }

  // Start a new operation. The reactor operation will be performed when the
  // given descriptor is flagged as ready, or an error has occurred.
  void start_op(int op_type, socket_type descriptor,
      per_descriptor_data& descriptor_data,
      reactor_op* op, bool allow_speculative)
  {
    mutex::scoped_lock descriptor_lock(descriptor_data->mutex_);
    if (descriptor_data->shutdown_)
      return;

    if (descriptor_data->op_queue_[op_type].empty())
    {
      if (allow_speculative
          && (op_type != read_op
            || descriptor_data->op_queue_[except_op].empty()))
      {
        if (op->perform())
        {
          descriptor_lock.unlock();
          io_service_.post_immediate_completion(op);
          return;
        }
      }
      else
      {
        epoll_event ev = { 0, { 0 } };
        ev.events = EPOLLIN | EPOLLERR | EPOLLHUP
          | EPOLLOUT | EPOLLPRI | EPOLLET;
        ev.data.ptr = descriptor_data;
        epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
      }
    }

    descriptor_data->op_queue_[op_type].push(op);
    io_service_.work_started();
  }

  // Cancel all operations associated with the given descriptor. The
  // handlers associated with the descriptor will be invoked with the
  // operation_aborted error.
  void cancel_ops(socket_type descriptor, per_descriptor_data& descriptor_data)
  {
    mutex::scoped_lock descriptor_lock(descriptor_data->mutex_);

    op_queue<operation> ops;
    for (int i = 0; i < max_ops; ++i)
    {
      while (reactor_op* op = descriptor_data->op_queue_[i].front())
      {
        op->ec_ = asio::error::operation_aborted;
        descriptor_data->op_queue_[i].pop();
        ops.push(op);
      }
    }

    descriptor_lock.unlock();

    io_service_.post_deferred_completions(ops);
  }

  // Cancel any operations that are running against the descriptor and remove
  // its registration from the reactor.
  void close_descriptor(socket_type descriptor,
      per_descriptor_data& descriptor_data)
  {
    mutex::scoped_lock descriptor_lock(descriptor_data->mutex_);
    mutex::scoped_lock descriptors_lock(registered_descriptors_mutex_);

    // Remove the descriptor from the set of known descriptors. The descriptor
    // will be automatically removed from the epoll set when it is closed.
    descriptor_data->shutdown_ = true;

    op_queue<operation> ops;
    for (int i = 0; i < max_ops; ++i)
    {
      while (reactor_op* op = descriptor_data->op_queue_[i].front())
      {
        op->ec_ = asio::error::operation_aborted;
        descriptor_data->op_queue_[i].pop();
        ops.push(op);
      }
    }

    descriptor_lock.unlock();

    registered_descriptors_.erase(descriptor);

    descriptors_lock.unlock();

    io_service_.post_deferred_completions(ops);
  }

  // Add a new timer queue to the reactor.
  template <typename Time_Traits>
  void add_timer_queue(timer_queue<Time_Traits>& timer_queue)
  {
    mutex::scoped_lock lock(mutex_);
    timer_queues_.insert(&timer_queue);
  }

  // Remove a timer queue from the reactor.
  template <typename Time_Traits>
  void remove_timer_queue(timer_queue<Time_Traits>& timer_queue)
  {
    mutex::scoped_lock lock(mutex_);
    timer_queues_.erase(&timer_queue);
  }

  // Schedule a new operation in the given timer queue to expire at the
  // specified absolute time.
  template <typename Time_Traits>
  void schedule_timer(timer_queue<Time_Traits>& timer_queue,
      const typename Time_Traits::time_type& time, timer_op* op, void* token)
  {
    mutex::scoped_lock lock(mutex_);
    if (!shutdown_)
    {
      bool earliest = timer_queue.enqueue_timer(time, op, token);
      io_service_.work_started();
      if (earliest)
      {
#if defined(ASIO_HAS_TIMERFD)
        if (timer_fd_ != -1)
        {
          itimerspec new_timeout;
          itimerspec old_timeout;
          int flags = get_timeout(new_timeout);
          timerfd_settime(timer_fd_, flags, &new_timeout, &old_timeout);
          return;
        }
#endif // defined(ASIO_HAS_TIMERFD)
        interrupter_.interrupt();
      }
    }
  }

  // Cancel the timer operations associated with the given token. Returns the
  // number of operations that have been posted or dispatched.
  template <typename Time_Traits>
  std::size_t cancel_timer(timer_queue<Time_Traits>& timer_queue, void* token)
  {
    mutex::scoped_lock lock(mutex_);
    op_queue<operation> ops;
    std::size_t n = timer_queue.cancel_timer(token, ops);
    lock.unlock();
    io_service_.post_deferred_completions(ops);
    return n;
  }

  // Run epoll once until interrupted or events are ready to be dispatched.
  void run(bool block, op_queue<operation>& ops)
  {
    // Calculate a timeout only if timerfd is not used.
    int timeout;
    if (timer_fd_ != -1)
      timeout = block ? -1 : 0;
    else
    {
      mutex::scoped_lock lock(mutex_);
      timeout = block ? get_timeout() : 0;
    }

    // Block on the epoll descriptor.
    epoll_event events[128];
    int num_events = epoll_wait(epoll_fd_, events, 128, timeout);

#if defined(ASIO_HAS_TIMERFD)
    bool check_timers = (timer_fd_ == -1);
#else // defined(ASIO_HAS_TIMERFD)
    bool check_timers = true;
#endif // defined(ASIO_HAS_TIMERFD)

    // Dispatch the waiting events.
    for (int i = 0; i < num_events; ++i)
    {
      void* ptr = events[i].data.ptr;
      if (ptr == &interrupter_)
      {
        // No need to reset the interrupter since we're leaving the descriptor
        // in a ready-to-read state and relying on edge-triggered notifications
        // to make it so that we only get woken up when the descriptor's epoll
        // registration is updated.

#if defined(ASIO_HAS_TIMERFD)
        if (timer_fd_ == -1)
          check_timers = true;
#else // defined(ASIO_HAS_TIMERFD)
        check_timers = true;
#endif // defined(ASIO_HAS_TIMERFD)
      }
#if defined(ASIO_HAS_TIMERFD)
      else if (ptr == &timer_fd_)
      {
        check_timers = true;
      }
#endif // defined(ASIO_HAS_TIMERFD)
      else
      {
        descriptor_state* descriptor_data = static_cast<descriptor_state*>(ptr);
        mutex::scoped_lock descriptor_lock(descriptor_data->mutex_);

        // Exception operations must be processed first to ensure that any
        // out-of-band data is read before normal data.
        static const int flag[max_ops] = { EPOLLIN, EPOLLOUT, EPOLLPRI };
        for (int j = max_ops - 1; j >= 0; --j)
        {
          if (events[i].events & (flag[j] | EPOLLERR | EPOLLHUP))
          {
            while (reactor_op* op = descriptor_data->op_queue_[j].front())
            {
              if (op->perform())
              {
                descriptor_data->op_queue_[j].pop();
                ops.push(op);
              }
              else
                break;
            }
          }
        }
      }
    }

    if (check_timers)
    {
      mutex::scoped_lock common_lock(mutex_);
      timer_queues_.get_ready_timers(ops);

#if defined(ASIO_HAS_TIMERFD)
      if (timer_fd_ != -1)
      {
        itimerspec new_timeout;
        itimerspec old_timeout;
        int flags = get_timeout(new_timeout);
        timerfd_settime(timer_fd_, flags, &new_timeout, &old_timeout);
      }
#endif // defined(ASIO_HAS_TIMERFD)
    }
  }

  // Interrupt the select loop.
  void interrupt()
  {
    epoll_event ev = { 0, { 0 } };
    ev.events = EPOLLIN | EPOLLERR | EPOLLET;
    ev.data.ptr = &interrupter_;
    epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, interrupter_.read_descriptor(), &ev);
  }

private:
  // The hint to pass to epoll_create to size its data structures.
  enum { epoll_size = 20000 };

  // Create the epoll file descriptor. Throws an exception if the descriptor
  // cannot be created.
  static int do_epoll_create()
  {
    int fd = epoll_create(epoll_size);
    if (fd == -1)
    {
      boost::throw_exception(
          asio::system_error(
            asio::error_code(errno,
              asio::error::get_system_category()),
            "epoll"));
    }
    return fd;
  }

  // Get the timeout value for the epoll_wait call. The timeout value is
  // returned as a number of milliseconds. A return value of -1 indicates
  // that epoll_wait should block indefinitely.
  int get_timeout()
  {
    // By default we will wait no longer than 5 minutes. This will ensure that
    // any changes to the system clock are detected after no longer than this.
    return timer_queues_.wait_duration_msec(5 * 60 * 1000);
  }

#if defined(ASIO_HAS_TIMERFD)
  // Get the timeout value for the timer descriptor. The return value is the
  // flag argument to be used when calling timerfd_settime.
  int get_timeout(itimerspec& ts)
  {
    ts.it_interval.tv_sec = 0;
    ts.it_interval.tv_nsec = 0;

    long usec = timer_queues_.wait_duration_usec(5 * 60 * 1000 * 1000);
    ts.it_value.tv_sec = usec / 1000000;
    ts.it_value.tv_nsec = usec ? (usec % 1000000) * 1000 : 1;

    return usec ? 0 : TFD_TIMER_ABSTIME;
  }
#endif // defined(ASIO_HAS_TIMERFD)

  // The io_service implementation used to post completions.
  io_service_impl& io_service_;

  // Mutex to protect access to internal data.
  mutex mutex_;

  // The epoll file descriptor.
  int epoll_fd_;

  // The timer file descriptor.
  int timer_fd_;

  // The interrupter is used to break a blocking epoll_wait call.
  select_interrupter interrupter_;

  // The timer queues.
  timer_queue_set timer_queues_;

  // Whether the service has been shut down.
  bool shutdown_;

  // Mutex to protect access to the registered descriptors.
  mutex registered_descriptors_mutex_;

  // Keep track of all registered descriptors. This code relies on the fact that
  // the hash_map implementation pools deleted nodes, meaning that we can assume
  // our descriptor_state pointer remains valid even after the entry is removed.
  // Technically this is not true for C++98, as that standard says that spliced
  // elements in a list are invalidated. However, C++0x fixes this shortcoming
  // so we'll just assume that C++98 std::list implementations will do the right
  // thing anyway.
  typedef detail::hash_map<socket_type, descriptor_state> descriptor_map;
  descriptor_map registered_descriptors_;
};

} // namespace detail
} // namespace asio

#endif // defined(ASIO_HAS_EPOLL)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_EPOLL_REACTOR_HPP
