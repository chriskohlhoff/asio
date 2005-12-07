//
// epoll_reactor.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <cstddef>
#include <sys/epoll.h>
#include <boost/config.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/system_exception.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/hash_map.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/task_demuxer_service.hpp"
#include "asio/detail/thread.hpp"
#include "asio/detail/reactor_op_queue.hpp"
#include "asio/detail/reactor_timer_queue.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/signal_blocker.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <bool Own_Thread>
class epoll_reactor
  : private noncopyable
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

  // Start a new read operation. The handler object will be invoked when the
  // given descriptor is ready to be read, or an error has occurred.
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
      if (except_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLPRI;
      ev.data.fd = descriptor;

      int result;
      if (epoll_registrations_.find(descriptor) == epoll_registrations_.end())
      {
        epoll_registrations_.insert(
            epoll_registration_map::value_type(descriptor, true));
        result = epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, descriptor, &ev);
      }
      else
      {
        result = epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
      }

      if (result != 0)
      {
        int error = errno;
        read_op_queue_.dispatch_all_operations(descriptor, error);
      }
    }
  }

  // Start a new write operation. The handler object will be invoked when the
  // given descriptor is ready to be written, or an error has occurred.
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
      if (except_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLPRI;
      ev.data.fd = descriptor;

      int result;
      if (epoll_registrations_.find(descriptor) == epoll_registrations_.end())
      {
        epoll_registrations_.insert(
            epoll_registration_map::value_type(descriptor, true));
        result = epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, descriptor, &ev);
      }
      else
      {
        result = epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
      }

      if (result != 0)
      {
        int error = errno;
        write_op_queue_.dispatch_all_operations(descriptor, error);
      }
    }
  }

  // Start a new exception operation. The handler object will be invoked when
  // the given descriptor has exception information, or an error has occurred.
  template <typename Handler>
  void start_except_op(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    if (except_op_queue_.enqueue_operation(descriptor, handler))
    {
      epoll_event ev = { 0 };
      ev.events = EPOLLPRI | EPOLLERR | EPOLLHUP;
      if (read_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLIN;
      if (write_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLOUT;
      ev.data.fd = descriptor;

      int result;
      if (epoll_registrations_.find(descriptor) == epoll_registrations_.end())
      {
        epoll_registrations_.insert(
            epoll_registration_map::value_type(descriptor, true));
        result = epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, descriptor, &ev);
      }
      else
      {
        result = epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
      }

      if (result != 0)
      {
        int error = errno;
        except_op_queue_.dispatch_all_operations(descriptor, error);
      }
    }
  }

  // Start new write and exception operations. The handler object will be
  // invoked when the given descriptor is ready for writing or has exception
  // information available, or an error has occurred.
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
      ev.events = EPOLLOUT | EPOLLPRI | EPOLLERR | EPOLLHUP;
      if (read_op_queue_.has_operation(descriptor))
        ev.events |= EPOLLIN;
      ev.data.fd = descriptor;

      int result;
      if (epoll_registrations_.find(descriptor) == epoll_registrations_.end())
      {
        epoll_registrations_.insert(
            epoll_registration_map::value_type(descriptor, true));
        result = epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, descriptor, &ev);
      }
      else
      {
        result = epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
      }

      if (result != 0)
      {
        int error = errno;
        write_op_queue_.dispatch_all_operations(descriptor, error);
        except_op_queue_.dispatch_all_operations(descriptor, error);
      }
    }
  }

  // Cancel all operations associated with the given descriptor. The
  // handlers associated with the descriptor will be invoked with the
  // operation_aborted error.
  void cancel_ops(socket_type descriptor)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    cancel_ops_unlocked(descriptor);
  }

  // Enqueue cancellation of all operations associated with the given
  // descriptor. The handlers associated with the descriptor will be invoked
  // with the operation_aborted error. This function does not acquire the
  // select_reactor's mutex, and so should only be used from within a reactor
  // handler.
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

  // Schedule a timer to expire at the specified absolute time. The handler
  // object will be invoked when the timer expires.
  template <typename Handler>
  void schedule_timer(const boost::posix_time::ptime& time,
      Handler handler, void* token)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (timer_queue_.enqueue_timer(time, handler, token))
      interrupter_.interrupt();
  }

  // Cancel the timer associated with the given token. Returns the number of
  // handlers that have been posted or dispatched.
  std::size_t cancel_timer(void* token)
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
    while (!stop && !stop_thread_)
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
            except_op_queue_.dispatch_all_operations(descriptor, 0);
            read_op_queue_.dispatch_all_operations(descriptor, 0);
            write_op_queue_.dispatch_all_operations(descriptor, 0);

            epoll_event ev = { 0 };
            ev.events = 0;
            ev.data.fd = descriptor;
            epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
          }
          else
          {
            bool more_reads = false;
            bool more_writes = false;
            bool more_except = false;

            // Exception operations must be processed first to ensure that any
            // out-of-band data is read before normal data.
            if (events[i].events & EPOLLPRI)
              more_except = except_op_queue_.dispatch_operation(descriptor, 0);
            else
              more_except = except_op_queue_.has_operation(descriptor);

            if (events[i].events & EPOLLIN)
              more_reads = read_op_queue_.dispatch_operation(descriptor, 0);
            else
              more_reads = read_op_queue_.has_operation(descriptor);

            if (events[i].events & EPOLLOUT)
              more_writes = write_op_queue_.dispatch_operation(descriptor, 0);
            else
              more_writes = write_op_queue_.has_operation(descriptor);

            epoll_event ev = { 0 };
            ev.events = EPOLLERR | EPOLLHUP;
            if (more_reads)
              ev.events |= EPOLLIN;
            if (more_writes)
              ev.events |= EPOLLOUT;
            if (more_except)
              ev.events |= EPOLLPRI;
            ev.data.fd = descriptor;
            int result = epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, descriptor, &ev);
            if (result != 0)
            {
              int error = errno;
              read_op_queue_.dispatch_all_operations(descriptor, error);
              write_op_queue_.dispatch_all_operations(descriptor, error);
              except_op_queue_.dispatch_all_operations(descriptor, error);
            }
          }
        }
      }
      read_op_queue_.dispatch_cancellations();
      write_op_queue_.dispatch_cancellations();
      except_op_queue_.dispatch_cancellations();
      timer_queue_.dispatch_timers(
          boost::posix_time::microsec_clock::universal_time());

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
    {
      system_exception e("epoll", errno);
      boost::throw_exception(e);
    }
    return fd;
  }

  // Get the timeout value for the epoll_wait call. The timeout value is
  // returned as a number of milliseconds. A return value of -1 indicates
  // that epoll_wait should block indefinitely.
  int get_timeout()
  {
    if (timer_queue_.empty())
      return -1;

    boost::posix_time::ptime now
      = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::ptime earliest_timer;
    timer_queue_.get_earliest_time(earliest_timer);
    if (now < earliest_timer)
    {
      boost::posix_time::time_duration timeout = earliest_timer - now;
      const int max_timeout_in_seconds = INT_MAX / 1000;
      if (max_timeout_in_seconds < timeout.total_seconds())
        return max_timeout_in_seconds * 1000;
      else
        return timeout.total_milliseconds();
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
  reactor_timer_queue<boost::posix_time::ptime> timer_queue_;

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
