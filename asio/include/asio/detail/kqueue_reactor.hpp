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
  enum op_types { read_op = 0, write_op = 1,
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
  kqueue_reactor(asio::io_service& io_service)
    : asio::detail::service_base<kqueue_reactor>(io_service),
      io_service_(use_service<io_service_impl>(io_service)),
      mutex_(),
      kqueue_fd_(do_kqueue_create()),
      interrupter_(),
      shutdown_(false)
  {
    // The interrupter is put into a permanently readable state. Whenever we
    // want to interrupt the blocked kevent call we register a one-shot read
    // operation against the descriptor.
    interrupter_.interrupt();
  }

  // Destructor.
  ~kqueue_reactor()
  {
    close(kqueue_fd_);
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

    bool first = descriptor_data->op_queue_[op_type].empty();
    if (first)
    {
      if (allow_speculative)
      {
        if (op_type != read_op || descriptor_data->op_queue_[except_op].empty())
        {
          if (op->perform())
          {
            descriptor_lock.unlock();
            io_service_.post_immediate_completion(op);
            return;
          }
        }
      }
    }

    descriptor_data->op_queue_[op_type].push(op);
    io_service_.work_started();

    if (first)
    {
      struct kevent event;
      switch (op_type)
      {
      case read_op:
        EV_SET(&event, descriptor, EVFILT_READ,
            EV_ADD | EV_ONESHOT, 0, 0, descriptor_data);
        break;
      case write_op:
        EV_SET(&event, descriptor, EVFILT_WRITE,
            EV_ADD | EV_ONESHOT, 0, 0, descriptor_data);
        break;
      case except_op:
        if (!descriptor_data->op_queue_[read_op].empty())
          return; // Already registered for read events.
        EV_SET(&event, descriptor, EVFILT_READ,
            EV_ADD | EV_ONESHOT, EV_OOBAND, 0, descriptor_data);
        break;
      }

      if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
      {
        op->ec_ = asio::error_code(errno,
            asio::error::get_system_category());
        descriptor_data->op_queue_[op_type].pop();
        io_service_.post_deferred_completion(op);
      }
    }
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
    // will be automatically removed from the kqueue set when it is closed.
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
        interrupt();
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

  // Run the kqueue loop.
  void run(bool block, op_queue<operation>& ops)
  {
    mutex::scoped_lock lock(mutex_);

    // Determine how long to block while waiting for events.
    timespec timeout_buf = { 0, 0 };
    timespec* timeout = block ? get_timeout(timeout_buf) : &timeout_buf;

    lock.unlock();

    // Block on the kqueue descriptor.
    struct kevent events[128];
    int num_events = kevent(kqueue_fd_, 0, 0, events, 128, timeout);

    // Dispatch the waiting events.
    for (int i = 0; i < num_events; ++i)
    {
      int descriptor = events[i].ident;
      void* ptr = events[i].udata;
      if (ptr == &interrupter_)
      {
        // No need to reset the interrupter since we're leaving the descriptor
        // in a ready-to-read state and relying on one-shot notifications.
      }
      else
      {
        descriptor_state* descriptor_data = static_cast<descriptor_state*>(ptr);
        mutex::scoped_lock descriptor_lock(descriptor_data->mutex_);

        // Exception operations must be processed first to ensure that any
        // out-of-band data is read before normal data.
        static const int filter[max_ops] =
          { EVFILT_READ, EVFILT_WRITE, EVFILT_READ };
        for (int j = max_ops - 1; j >= 0; --j)
        {
          if (events[i].filter == filter[j])
          {
            if (j != except_op || events[i].flags & EV_OOBAND)
            {
              while (reactor_op* op = descriptor_data->op_queue_[j].front())
              {
                if (events[i].flags & EV_ERROR)
                {
                  op->ec_ = asio::error_code(events[i].data,
                      asio::error::get_system_category());
                  descriptor_data->op_queue_[j].pop();
                  ops.push(op);
                }
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

        // Renew registration for event notifications.
        struct kevent event;
        switch (events[i].filter)
        {
        case EVFILT_READ:
          if (!descriptor_data->op_queue_[read_op].empty())
            EV_SET(&event, descriptor, EVFILT_READ,
                EV_ADD | EV_ONESHOT, 0, 0, descriptor_data);
          else if (!descriptor_data->op_queue_[except_op].empty())
            EV_SET(&event, descriptor, EVFILT_READ,
                EV_ADD | EV_ONESHOT, EV_OOBAND, 0, descriptor_data);
          else
            continue;
        case EVFILT_WRITE:
          if (!descriptor_data->op_queue_[write_op].empty())
            EV_SET(&event, descriptor, EVFILT_WRITE,
                EV_ADD | EV_ONESHOT, 0, 0, descriptor_data);
          else
            continue;
        default:
          break;
        }
        if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
        {
          asio::error_code error(errno,
              asio::error::get_system_category());
          for (int j = 0; j < max_ops; ++j)
          {
            while (reactor_op* op = descriptor_data->op_queue_[j].front())
            {
              op->ec_ = error;
              descriptor_data->op_queue_[j].pop();
              ops.push(op);
            }
          }
        }
      }
    }

    lock.lock();
    timer_queues_.get_ready_timers(ops);
  }

  // Interrupt the kqueue loop.
  void interrupt()
  {
    struct kevent event;
    EV_SET(&event, interrupter_.read_descriptor(),
        EVFILT_READ, EV_ADD | EV_ONESHOT, 0, 0, &interrupter_);
    ::kevent(kqueue_fd_, &event, 1, 0, 0, 0);
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

  // The io_service implementation used to post completions.
  io_service_impl& io_service_;

  // Mutex to protect access to internal data.
  mutex mutex_;

  // The kqueue file descriptor.
  int kqueue_fd_;

  // The interrupter is used to break a blocking kevent call.
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

#endif // defined(ASIO_HAS_KQUEUE)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_KQUEUE_REACTOR_HPP
