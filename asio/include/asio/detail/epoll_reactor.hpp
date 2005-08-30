//
// epoll_reactor.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
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

#if defined(__linux__) // This service is only supported on Linux.

#include "asio/detail/push_options.hpp"
#include <linux/version.h>
#include "asio/detail/pop_options.hpp"

#if LINUX_VERSION_CODE >= KERNEL_VERSION (2,5,45) // Only kernels >= 2.5.45.

// Define this to indicate that epoll is supported on the target platform.
#define ASIO_HAS_EPOLL_REACTOR 1

#include "asio/detail/push_options.hpp"
#include <new>
#include <sys/epoll.h>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/bind_handler.hpp"
#include "asio/detail/hash_map.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/task_demuxer_service.hpp"
#include "asio/detail/thread.hpp"
#include "asio/detail/reactor_op_queue.hpp"
#include "asio/detail/reactor_timer_queue.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/signal_blocker.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/time.hpp"

namespace asio {
namespace detail {

template <bool Own_Thread>
class epoll_reactor
  : private boost::noncopyable
{
public:
  // Constructor.
  template <typename Demuxer>
  epoll_reactor(Demuxer&)
    : mutex_(),
      epoll_fd_(do_epoll_create()),
      wait_in_progress_(false),
      interrupter_(),
      read_op_queue_(),
      write_op_queue_(),
      except_op_queue_(),
      epoll_registrations_(),
      pending_cancellations_(),
      stop_thread_(false),
      thread_(0)
  {
    // Start the reactor's internal thread only if needed.
    if (Own_Thread)
    {
      asio::detail::signal_blocker sb;
      thread_ = new asio::detail::thread(
          bind_handler(&epoll_reactor::call_run_thread, this));
    }

    // Add the interrupter's descriptor to epoll.
    epoll_event ev = { 0 };
    ev.events = EPOLLIN | EPOLLERR;
    ev.data.fd = interrupter_.read_descriptor();
    epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, interrupter_.read_descriptor(), &ev);
  }

  // Destructor.
  ~epoll_reactor()
  {
    if (thread_)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      stop_thread_ = true;
      lock.unlock();
      interrupter_.interrupt();
      thread_->join();
      delete thread_;
    }

    close(epoll_fd_);
  }

  // Start a new read operation. The do_operation function of the select_op
  // object will be invoked when the given descriptor is ready to be read.
  template <typename Handler>
  void start_read_op(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    if (read_op_queue_.enqueue_operation(descriptor, handler))
    {
      epoll_event ev = { 0 };
      ev.events = EPOLLIN | EPOLLERR | EPOLLHUP;
      if (write_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLOUT;
      ev.data.fd = descriptor;

      if (epoll_registrations_.find(descriptor) == epoll_registrations_.end())
      {
        epoll_registrations_.insert(
            epoll_registration_map::value_type(descriptor, true));
        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, descriptor, &ev);
      }
      else
      {
        epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
      }
    }
  }

  // Start a new write operation. The do_operation function of the select_op
  // object will be invoked when the given descriptor is ready for writing.
  template <typename Handler>
  void start_write_op(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    if (write_op_queue_.enqueue_operation(descriptor, handler))
    {
      epoll_event ev = { 0 };
      ev.events = EPOLLOUT | EPOLLERR | EPOLLHUP;
      if (read_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLIN;
      ev.data.fd = descriptor;

      if (epoll_registrations_.find(descriptor) == epoll_registrations_.end())
      {
        epoll_registrations_.insert(
            epoll_registration_map::value_type(descriptor, true));
        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, descriptor, &ev);
      }
      else
      {
        epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
      }
    }
  }

  // Start a new exception operation. The do_operation function of the select_op
  // object will be invoked when the given descriptor has exception information
  // available.
  template <typename Handler>
  void start_except_op(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    if (except_op_queue_.enqueue_operation(descriptor, handler))
    {
      epoll_event ev = { 0 };
      ev.events = EPOLLERR | EPOLLHUP;
      if (read_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLIN;
      if (write_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLOUT;
      ev.data.fd = descriptor;

      if (epoll_registrations_.find(descriptor) == epoll_registrations_.end())
      {
        epoll_registrations_.insert(
            epoll_registration_map::value_type(descriptor, true));
        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, descriptor, &ev);
      }
      else
      {
        epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
      }
    }
  }

  // Start a new write and exception operations. The do_operation function of
  // the select_op object will be invoked when the given descriptor is ready
  // for writing or has exception information available.
  template <typename Handler>
  void start_write_and_except_ops(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    bool need_mod = write_op_queue_.enqueue_operation(descriptor, handler);
    need_mod = except_op_queue_.enqueue_operation(descriptor, handler)
      && need_mod;
    if (need_mod)
    {
      epoll_event ev = { 0 };
      ev.events = EPOLLOUT | EPOLLERR | EPOLLHUP;
      if (read_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLIN;
      ev.data.fd = descriptor;

      if (epoll_registrations_.find(descriptor) == epoll_registrations_.end())
      {
        epoll_registrations_.insert(
            epoll_registration_map::value_type(descriptor, true));
        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, descriptor, &ev);
      }
      else
      {
        epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
      }
    }
  }

  // Cancel all operations associated with the given descriptor. The
  // do_cancel function of the handler objects will be invoked.
  void cancel_ops(socket_type descriptor)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    cancel_ops_unlocked(descriptor);
  }

  // Enqueue cancellation of all operations associated with the given
  // descriptor. The do_cancel function of the handler objects will be invoked.
  // This function does not acquire the epoll_reactor's mutex, and so should
  // only be used from within a reactor handler.
  void enqueue_cancel_ops_unlocked(socket_type descriptor)
  {
    pending_cancellations_.insert(
        pending_cancellations_map::value_type(descriptor, true));
  }

  // Cancel any operations that are running against the descriptor and remove
  // its registration from the reactor.
  void close_descriptor(socket_type descriptor)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // Remove the descriptor from epoll.
    epoll_registration_map::iterator it = epoll_registrations_.find(descriptor);
    if (it != epoll_registrations_.end())
    {
      epoll_event ev = { 0 };
      epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, descriptor, &ev);
      epoll_registrations_.erase(it);
    }

    // Cancel any outstanding operations associated with the descriptor.
    cancel_ops_unlocked(descriptor);
  }

  // Schedule a timer to expire at the specified absolute time. The
  // do_operation function of the handler object will be invoked when the timer
  // expires. Returns a token that may be used for cancelling the timer, but it
  // is not valid after the timer expires.
  template <typename Handler>
  void schedule_timer(long sec, long usec, Handler handler, void* token)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (timer_queue_.enqueue_timer(detail::time(sec, usec), handler, token))
      interrupter_.interrupt();
  }

  // Cancel the timer associated with the given token. Returns the number of
  // handlers that have been posted or dispatched.
  int cancel_timer(void* token)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    return timer_queue_.cancel_timer(token);
  }

private:
  friend class task_demuxer_service<epoll_reactor<Own_Thread> >;

  // Reset the select loop before a new run.
  void reset()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    stop_thread_ = false;
    interrupter_.reset();
  }

  // Run the epoll loop.
  void run()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // Dispatch any operation cancellations that were made while the select
    // loop was not running.
    read_op_queue_.dispatch_cancellations();
    write_op_queue_.dispatch_cancellations();
    except_op_queue_.dispatch_cancellations();

    bool stop = false;
    while (!stop)
    {
      int timeout = get_timeout();
      wait_in_progress_ = true;
      lock.unlock();

      // Block on the epoll descriptor.
      epoll_event events[128];
      int num_events = epoll_wait(epoll_fd_, events, 128, timeout);

      lock.lock();
      wait_in_progress_ = false;

      // Block signals while dispatching operations.
      asio::detail::signal_blocker sb;

      // Dispatch the waiting events.
      for (int i = 0; i < num_events; ++i)
      {
        int descriptor = events[i].data.fd;
        if (descriptor == interrupter_.read_descriptor())
        {
          stop = interrupter_.reset();
        }
        else
        {
          if (events[i].events & (EPOLLERR | EPOLLHUP))
          {
            except_op_queue_.dispatch_all_operations(descriptor);
            read_op_queue_.dispatch_all_operations(descriptor);
            write_op_queue_.dispatch_all_operations(descriptor);

            epoll_event ev = { 0 };
            ev.events = 0;
            ev.data.fd = descriptor;
            epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
          }
          else
          {
            bool more_reads = false;
            bool more_writes = false;

            if (events[i].events & EPOLLIN)
              more_reads = read_op_queue_.dispatch_operation(descriptor);
            else
              more_reads = read_op_queue_.has_operation(descriptor);

            if (events[i].events & EPOLLOUT)
              more_writes = write_op_queue_.dispatch_operation(descriptor);
            else
              more_writes = write_op_queue_.has_operation(descriptor);

            epoll_event ev = { 0 };
            ev.events = EPOLLERR | EPOLLHUP;
            if (more_reads)
              ev.events |= EPOLLIN;
            if (more_writes)
              ev.events |= EPOLLOUT;
            ev.data.fd = descriptor;
            epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
          }
        }
      }
      read_op_queue_.dispatch_cancellations();
      write_op_queue_.dispatch_cancellations();
      except_op_queue_.dispatch_cancellations();
      timer_queue_.dispatch_timers(detail::time::now());

      // Issue any pending cancellations.
      pending_cancellations_map::iterator i = pending_cancellations_.begin();
      while (i != pending_cancellations_.end())
      {
        cancel_ops_unlocked(i->first);
        ++i;
      }
      pending_cancellations_.clear();
    }
  }

  // Run the select loop in the thread.
  void run_thread()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    while (!stop_thread_)
    {
      lock.unlock();
      run();
      lock.lock();
    }
  }

  // Entry point for the select loop thread.
  static void call_run_thread(epoll_reactor* reactor)
  {
    reactor->run_thread();
  }

  // Interrupt the select loop.
  void interrupt()
  {
    interrupter_.interrupt();
  }

  // The hint to pass to epoll_create to size its data structures.
  enum { epoll_size = 20000 };

  // Create the epoll file descriptor. Throws an exception if the descriptor
  // cannot be created.
  static int do_epoll_create()
  {
    int fd = epoll_create(epoll_size);
    if (fd == -1)
      throw std::bad_alloc();
    return fd;
  }

  // Get the timeout value for the epoll_wait call. The timeout value is
  // returned as a number of milliseconds. A return value of -1 indicates
  // that epoll_wait should block indefinitely.
  int get_timeout()
  {
    if (timer_queue_.empty())
      return -1;

    detail::time now = detail::time::now();
    detail::time earliest_timer;
    timer_queue_.get_earliest_time(earliest_timer);
    if (now < earliest_timer)
    {
      detail::time timeout = earliest_timer;
      timeout -= now;
      const int max_timeout_in_seconds = INT_MAX / 1000;
      if (max_timeout_in_seconds < timeout)
        return max_timeout_in_seconds * 1000;
      else
        return timeout.sec() * 1000 + timeout.usec() / 1000;
    }
    else
    {
      return 0;
    }
  }

  // Cancel all operations associated with the given descriptor. The do_cancel
  // function of the handler objects will be invoked. This function does not
  // acquire the epoll_reactor's mutex.
  void cancel_ops_unlocked(socket_type descriptor)
  {
    bool interrupt = read_op_queue_.cancel_operations(descriptor);
    interrupt = write_op_queue_.cancel_operations(descriptor) || interrupt;
    interrupt = except_op_queue_.cancel_operations(descriptor) || interrupt;
    if (interrupt)
      interrupter_.interrupt();
  }

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The epoll file descriptor.
  int epoll_fd_;

  // Whether the epoll_wait call is currently in progress
  bool wait_in_progress_;

  // The interrupter is used to break a blocking epoll_wait call.
  select_interrupter interrupter_;

  // The queue of read operations.
  reactor_op_queue<socket_type> read_op_queue_;

  // The queue of write operations.
  reactor_op_queue<socket_type> write_op_queue_;

  // The queue of except operations.
  reactor_op_queue<socket_type> except_op_queue_;

  // The queue of timers.
  reactor_timer_queue<detail::time> timer_queue_;

  // The type for a map of descriptors that are registered with epoll.
  typedef hash_map<socket_type, bool> epoll_registration_map;

  // The map of descriptors that are registered with epoll.
  epoll_registration_map epoll_registrations_;

  // The type for a map of descriptors to be cancelled.
  typedef hash_map<socket_type, bool> pending_cancellations_map;

  // The map of descriptors that are pending cancellation.
  pending_cancellations_map pending_cancellations_;

  // Does the reactor loop thread need to stop.
  bool stop_thread_;

  // The thread that is running the reactor loop.
  asio::detail::thread* thread_;
};

} // namespace detail
} // namespace asio

#endif //  LINUX_VERSION_CODE >= KERNEL_VERSION (2,5,45)
#endif // __linux__

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_EPOLL_REACTOR_HPP
