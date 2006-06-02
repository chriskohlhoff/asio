//
// kqueue_reactor.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <vector>
#include <sys/types.h>
#include <sys/event.h>
#include <sys/time.h>
#include <boost/config.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/io_service.hpp"
#include "asio/system_exception.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/task_io_service.hpp"
#include "asio/detail/thread.hpp"
#include "asio/detail/reactor_op_queue.hpp"
#include "asio/detail/reactor_timer_queue.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/signal_blocker.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <bool Own_Thread>
class kqueue_reactor
  : public asio::io_service::service
{
public:
  // Constructor.
  kqueue_reactor(asio::io_service& io_service)
    : asio::io_service::service(io_service),
      mutex_(),
      kqueue_fd_(do_kqueue_create()),
      wait_in_progress_(false),
      interrupter_(),
      read_op_queue_(),
      write_op_queue_(),
      except_op_queue_(),
      pending_cancellations_(),
      stop_thread_(false),
      thread_(0),
      shutdown_(false)
  {
    // Start the reactor's internal thread only if needed.
    if (Own_Thread)
    {
      asio::detail::signal_blocker sb;
      thread_ = new asio::detail::thread(
          bind_handler(&kqueue_reactor::call_run_thread, this));
    }

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
    stop_thread_ = true;
    lock.unlock();

    if (thread_)
    {
      interrupter_.interrupt();
      thread_->join();
      delete thread_;
      thread_ = 0;
    }

    read_op_queue_.destroy_operations();
    write_op_queue_.destroy_operations();
    except_op_queue_.destroy_operations();
    timer_queue_.destroy_timers();
  }

  // Register a socket with the reactor. Returns 0 on success, system error
  // code on failure.
  int register_descriptor(socket_type)
  {
    return 0;
  }

  // Start a new read operation. The handler object will be invoked when the
  // given descriptor is ready to be read, or an error has occurred.
  template <typename Handler>
  void start_read_op(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    if (shutdown_)
      return;

    if (!read_op_queue_.has_operation(descriptor))
      if (handler(0))
        return;

    if (read_op_queue_.enqueue_operation(descriptor, handler))
    {
      struct kevent event;
      EV_SET(&event, descriptor, EVFILT_READ, EV_ADD, 0, 0, 0);
      if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
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

    if (shutdown_)
      return;

    if (!write_op_queue_.has_operation(descriptor))
      if (handler(0))
        return;

    if (write_op_queue_.enqueue_operation(descriptor, handler))
    {
      struct kevent event;
      EV_SET(&event, descriptor, EVFILT_WRITE, EV_ADD, 0, 0, 0);
      if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
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

    if (shutdown_)
      return;

    if (except_op_queue_.enqueue_operation(descriptor, handler))
    {
      struct kevent event;
      if (read_op_queue_.has_operation(descriptor))
        EV_SET(&event, descriptor, EVFILT_READ, EV_ADD, 0, 0, 0);
      else
        EV_SET(&event, descriptor, EVFILT_READ, EV_ADD, EV_OOBAND, 0, 0);
      if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
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

    if (shutdown_)
      return;

    if (write_op_queue_.enqueue_operation(descriptor, handler))
    {
      struct kevent event;
      EV_SET(&event, descriptor, EVFILT_WRITE, EV_ADD, 0, 0, 0);
      if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
      {
        int error = errno;
        write_op_queue_.dispatch_all_operations(descriptor, error);
      }
    }

    if (except_op_queue_.enqueue_operation(descriptor, handler))
    {
      struct kevent event;
      if (read_op_queue_.has_operation(descriptor))
        EV_SET(&event, descriptor, EVFILT_READ, EV_ADD, 0, 0, 0);
      else
        EV_SET(&event, descriptor, EVFILT_READ, EV_ADD, EV_OOBAND, 0, 0);
      if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
      {
        int error = errno;
        except_op_queue_.dispatch_all_operations(descriptor, error);
        write_op_queue_.dispatch_all_operations(descriptor, error);
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
  // kqueue_reactor's mutex, and so should only be used from within a reactor
  // handler.
  void enqueue_cancel_ops_unlocked(socket_type descriptor)
  {
    pending_cancellations_.push_back(descriptor);
  }

  // Cancel any operations that are running against the descriptor and remove
  // its registration from the reactor.
  void close_descriptor(socket_type descriptor)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // Remove the descriptor from kqueue.
    struct kevent event[2];
    EV_SET(&event[0], descriptor, EVFILT_READ, EV_DELETE, 0, 0, 0);
    EV_SET(&event[1], descriptor, EVFILT_WRITE, EV_DELETE, 0, 0, 0);
    ::kevent(kqueue_fd_, event, 2, 0, 0, 0);
    
    // Cancel any outstanding operations associated with the descriptor.
    cancel_ops_unlocked(descriptor);
  }

  // Schedule a timer to expire at the specified absolute time. The
  // do_operation function of the handler object will be invoked when the timer
  // expires. Returns a token that may be used for cancelling the timer, but it
  // is not valid after the timer expires.
  template <typename Handler>
  void schedule_timer(const boost::posix_time::ptime& time,
      Handler handler, void* token)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (!shutdown_)
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
  friend class task_io_service<kqueue_reactor<Own_Thread> >;

  // Run the kqueue loop.
  void run(bool block)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // Dispatch any operation cancellations that were made while the select
    // loop was not running.
    read_op_queue_.dispatch_cancellations();
    write_op_queue_.dispatch_cancellations();
    except_op_queue_.dispatch_cancellations();

    // Check if the thread is supposed to stop.
    if (stop_thread_)
    {
      // Clean up operations. We must not hold the lock since the operations may
      // make calls back into this reactor.
      lock.unlock();
      read_op_queue_.cleanup_operations();
      write_op_queue_.cleanup_operations();
      except_op_queue_.cleanup_operations();
      return;
    }

    // We can return immediately if there's no work to do and the reactor is
    // not supposed to block.
    if (!block && read_op_queue_.empty() && write_op_queue_.empty()
        && except_op_queue_.empty() && timer_queue_.empty())
    {
      // Clean up operations. We must not hold the lock since the operations may
      // make calls back into this reactor.
      lock.unlock();
      read_op_queue_.cleanup_operations();
      write_op_queue_.cleanup_operations();
      except_op_queue_.cleanup_operations();
      return;
    }

    // Determine how long to block while waiting for events.
    timespec timeout_buf = { 0, 0 };
    timespec* timeout = block ? get_timeout(timeout_buf) : &timeout_buf;

    wait_in_progress_ = true;
    lock.unlock();

    // Block on the kqueue descriptor.
    struct kevent events[128];
    int num_events = kevent(kqueue_fd_, 0, 0, events, 128, timeout);

    lock.lock();
    wait_in_progress_ = false;

    // Block signals while dispatching operations.
    asio::detail::signal_blocker sb;

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
          int error = events[i].data;
          except_op_queue_.dispatch_all_operations(descriptor, error);
          read_op_queue_.dispatch_all_operations(descriptor, error);
        }
        else if (events[i].flags & EV_OOBAND)
        {
          more_except = except_op_queue_.dispatch_operation(descriptor, 0);
          if (events[i].data > 0)
            more_reads = read_op_queue_.dispatch_operation(descriptor, 0);
          else
            more_reads = read_op_queue_.has_operation(descriptor);
        }
        else
        {
          more_reads = read_op_queue_.dispatch_operation(descriptor, 0);
          more_except = except_op_queue_.has_operation(descriptor);
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
          int error = errno;
          except_op_queue_.dispatch_all_operations(descriptor, error);
          read_op_queue_.dispatch_all_operations(descriptor, error);
        }
      }
      else if (events[i].filter == EVFILT_WRITE)
      {
        // Dispatch operations associated with the descriptor.
        bool more_writes = false;
        if (events[i].flags & EV_ERROR)
        {
          int error = events[i].data;
          write_op_queue_.dispatch_all_operations(descriptor, error);
        }
        else
        {
          more_writes = write_op_queue_.dispatch_operation(descriptor, 0);
        }

        // Update the descriptor in the kqueue.
        struct kevent event;
        if (more_writes)
          EV_SET(&event, descriptor, EVFILT_WRITE, EV_ADD, 0, 0, 0);
        else
          EV_SET(&event, descriptor, EVFILT_WRITE, EV_DELETE, 0, 0, 0);
        if (::kevent(kqueue_fd_, &event, 1, 0, 0, 0) == -1)
        {
          int error = errno;
          write_op_queue_.dispatch_all_operations(descriptor, error);
        }
      }
    }

    read_op_queue_.dispatch_cancellations();
    write_op_queue_.dispatch_cancellations();
    except_op_queue_.dispatch_cancellations();
    timer_queue_.dispatch_timers(
        boost::posix_time::microsec_clock::universal_time());

    // Issue any pending cancellations.
    for (size_t i = 0; i < pending_cancellations_.size(); ++i)
      cancel_ops_unlocked(pending_cancellations_[i]);
    pending_cancellations_.clear();

    // Clean up operations. We must not hold the lock since the operations may
    // make calls back into this reactor.
    lock.unlock();
    read_op_queue_.cleanup_operations();
    write_op_queue_.cleanup_operations();
    except_op_queue_.cleanup_operations();
  }

  // Run the select loop in the thread.
  void run_thread()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    while (!stop_thread_)
    {
      lock.unlock();
      run(true);
      lock.lock();
    }
  }

  // Entry point for the select loop thread.
  static void call_run_thread(kqueue_reactor* reactor)
  {
    reactor->run_thread();
  }

  // Interrupt the select loop.
  void interrupt()
  {
    interrupter_.interrupt();
  }

  // Create the kqueue file descriptor. Throws an exception if the descriptor
  // cannot be created.
  static int do_kqueue_create()
  {
    int fd = kqueue();
    if (fd == -1)
    {
      system_exception e("kqueue", errno);
      boost::throw_exception(e);
    }
    return fd;
  }

  // Get the timeout value for the kevent call.
  timespec* get_timeout(timespec& ts)
  {
    if (timer_queue_.empty())
      return 0;

    boost::posix_time::ptime now
      = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::ptime earliest_timer;
    timer_queue_.get_earliest_time(earliest_timer);
    if (now < earliest_timer)
    {
      boost::posix_time::time_duration timeout = earliest_timer - now;
      ts.tv_sec = timeout.total_seconds();
      ts.tv_nsec = timeout.total_nanoseconds() % 1000000000;
    }
    else
    {
      ts.tv_sec = 0;
      ts.tv_nsec = 0;
    }

    return &ts;
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
  int kqueue_fd_;

  // Whether the kqueue wait call is currently in progress
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

  // The descriptors that are pending cancellation.
  std::vector<socket_type> pending_cancellations_;

  // Does the reactor loop thread need to stop.
  bool stop_thread_;

  // The thread that is running the reactor loop.
  asio::detail::thread* thread_;

  // Whether the service has been shut down.
  bool shutdown_;
};

} // namespace detail
} // namespace asio

#endif // defined(ASIO_HAS_KQUEUE)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_KQUEUE_REACTOR_HPP
