//
// select_reactor.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_SELECT_REACTOR_HPP
#define ASIO_DETAIL_SELECT_REACTOR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/task_demuxer_service.hpp"
#include "asio/detail/thread.hpp"
#include "asio/detail/reactor_op_queue.hpp"
#include "asio/detail/reactor_timer_queue.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/time.hpp"

namespace asio {
namespace detail {

class select_reactor
  : private boost::noncopyable
{
public:
  // Constructor.
  template <typename Demuxer>
  select_reactor(Demuxer&)
    : mutex_(),
      interrupter_(),
      read_op_queue_(),
      write_op_queue_(),
      stop_thread_(false),
      thread_(new asio::detail::thread(
            bind_handler(&select_reactor::call_run_thread, this)))
  {
  }

  // Constructor when running as a demuxer task.
  select_reactor(basic_demuxer<task_demuxer_service<select_reactor> >&)
    : mutex_(),
      interrupter_(),
      read_op_queue_(),
      write_op_queue_(),
      stop_thread_(false),
      thread_(0)
  {
  }

  // Destructor.
  ~select_reactor()
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
  }

  // Start a new read operation. The do_operation function of the select_op
  // object will be invoked when the given descriptor is ready to be read.
  template <typename Handler>
  void start_read_op(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (read_op_queue_.enqueue_operation(descriptor, handler))
      interrupter_.interrupt();
  }

  // Start a read operation from inside an op_call invocation. The do_operation
  // function of the handler object will be invoked when the given descriptor
  // is ready to be read.
  template <typename Handler>
  void restart_read_op(socket_type descriptor, Handler handler)
  {
    read_op_queue_.enqueue_operation(descriptor, handler);
  }

  // Start a new write operation. The do_operation function of the select_op
  // object will be invoked when the given descriptor is ready for writing.
  template <typename Handler>
  void start_write_op(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (write_op_queue_.enqueue_operation(descriptor, handler))
      interrupter_.interrupt();
  }

  // Start a write operation from inside an op_call invocation. The
  // do_operation function of the handler object will be invoked when the
  // given descriptor is ready for writing.
  template <typename Handler>
  void restart_write_op(socket_type descriptor, Handler handler)
  {
    write_op_queue_.enqueue_operation(descriptor, handler);
  }

  // Close the given descriptor and cancel any operations that are running
  // against it.
  void close_descriptor(socket_type descriptor)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    read_op_queue_.close_descriptor(descriptor);
    write_op_queue_.close_descriptor(descriptor);
  }

  // Schedule a timer to expire at the specified absolute time. The
  // do_operation function of the handler object will be invoked when the timer
  // expires. Returns a token that may be used for cancelling the timer, but it
  // is not valid after the timer expires.
  template <typename Handler>
  void schedule_timer(long sec, long usec, Handler handler, void*& token)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (timer_queue_.enqueue_timer(time(sec, usec), handler, token))
      interrupter_.interrupt();
  }

  // Cancel the timer associated with the given token.
  void expire_timer(void*& token)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (token)
    {
      timer_queue_.cancel_timer(token);
      token = 0;
    }
  }

private:
  friend class task_demuxer_service<select_reactor>;

  // Reset the select loop before a new run.
  void reset()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    stop_thread_ = false;
    interrupter_.reset();
  }

  // Run the select loop.
  void run()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    bool stop = false;
    while (!stop)
    {
      // Set up the descriptor sets.
      fd_set_adaptor read_fds;
      read_fds.set(interrupter_.read_descriptor());
      read_op_queue_.get_descriptors(read_fds);
      fd_set_adaptor write_fds;
      write_op_queue_.get_descriptors(write_fds);
      int max_fd = (read_fds.max_descriptor() > write_fds.max_descriptor()
          ? read_fds.max_descriptor() : write_fds.max_descriptor());

      // Block on the select call without holding the lock so that new
      // operations can be started while the call is executing.
      timeval tv_buf;
      timeval* tv = get_timeout(tv_buf);
      lock.unlock();
      int retval = socket_ops::select(max_fd + 1, read_fds, write_fds, 0, tv);
      lock.lock();

      // Reset the interrupter.
      if (read_fds.is_set(interrupter_.read_descriptor()))
        stop = interrupter_.reset();

      // Dispatch all ready operations.
      if (retval > 0)
      {
        read_op_queue_.dispatch_descriptors(read_fds);
        write_op_queue_.dispatch_descriptors(write_fds);
      }
      timer_queue_.dispatch_timers(time::now());
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
  static void call_run_thread(select_reactor* reactor)
  {
    reactor->run_thread();
  }

  // Interrupt the select loop.
  void interrupt()
  {
    interrupter_.interrupt();
  }

  // Get the timeout value for the select call.
  timeval* get_timeout(timeval& tv)
  {
    if (timer_queue_.empty())
      return 0;

    time now = time::now();
    time earliest_timer;
    timer_queue_.get_earliest_time(earliest_timer);
    if (now < earliest_timer)
    {
      time timeout = earliest_timer;
      timeout -= now;
      tv.tv_sec = timeout.sec();
      tv.tv_usec = timeout.usec();
    }
    else
    {
      tv.tv_sec = 0;
      tv.tv_usec = 0;
    }

    return &tv;
  }

  // Adapts the FD_SET type to meet the Descriptor_Set concept's requirements.
  class fd_set_adaptor
  {
  public:
    fd_set_adaptor()
      : max_descriptor_(-1)
    {
      FD_ZERO(&fd_set_);
    }

    void set(socket_type descriptor)
    {
      if (descriptor > max_descriptor_)
        max_descriptor_ = descriptor;
      FD_SET(descriptor, &fd_set_);
    }

    bool is_set(socket_type descriptor) const
    {
      return FD_ISSET(descriptor, &fd_set_);
    }

    operator fd_set*()
    {
      return &fd_set_;
    }

    socket_type max_descriptor() const
    {
      return max_descriptor_;
    }

  private:
    fd_set fd_set_;
    socket_type max_descriptor_;
  };

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The interrupter is used to break a blocking select call.
  select_interrupter interrupter_;

  // The queue of read operations.
  reactor_op_queue<socket_type> read_op_queue_;

  // The queue of write operations.
  reactor_op_queue<socket_type> write_op_queue_;

  // The queue of timers.
  reactor_timer_queue<time> timer_queue_;

  // Does the reactor loop thread need to stop.
  bool stop_thread_;

  // The thread that is running the reactor loop.
  asio::detail::thread* thread_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SELECT_REACTOR_HPP
