//
// select_reactor.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SELECT_REACTOR_HPP
#define ASIO_DETAIL_SELECT_REACTOR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/hash_map.hpp"
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
      select_in_progress_(false),
      interrupter_(),
      read_op_queue_(),
      write_op_queue_(),
      except_op_queue_(),
      pending_cancellations_(),
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
      except_op_queue_(),
      pending_cancellations_(),
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

  // Start a new write operation. The do_operation function of the select_op
  // object will be invoked when the given descriptor is ready for writing.
  template <typename Handler>
  void start_write_op(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (write_op_queue_.enqueue_operation(descriptor, handler))
      interrupter_.interrupt();
  }

  // Start a new exception operation. The do_operation function of the
  // select_op object will be invoked when the given descriptor has exception
  // information available.
  template <typename Handler>
  void start_except_op(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (except_op_queue_.enqueue_operation(descriptor, handler))
      interrupter_.interrupt();
  }

  // Start a new write and exception operations. The do_operation function of
  // the select_op object will be invoked when the given descriptor is ready
  // for writing or has exception information available.
  template <typename Handler>
  void start_write_and_except_ops(socket_type descriptor, Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    bool interrupt = write_op_queue_.enqueue_operation(descriptor, handler);
    interrupt = except_op_queue_.enqueue_operation(descriptor, handler)
      || interrupt;
    if (interrupt)
      interrupter_.interrupt();
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
  // This function does not acquire the select_reactor's mutex, and so should
  // only be used from within a reactor handler.
  void enqueue_cancel_ops_unlocked(socket_type descriptor)
  {
    pending_cancellations_.insert(
        pending_cancellations_map::value_type(descriptor, true));
  }

  // Class template to adapt a close function as a timer handler.
  template <typename Close_Function>
  class close_handler
  {
  public:
    close_handler(socket_type descriptor, Close_Function close_function)
      : descriptor_(descriptor),
        close_function_(close_function)
    {
    }

    void do_operation()
    {
      close_function_(descriptor_);
    }

    void do_cancel()
    {
    }

  private:
    socket_type descriptor_;
    Close_Function close_function_;
  };

  // Close the given descriptor and cancel any operations that are running
  // against it. The given close function will be called to actually perform
  // the closure of the resource.
  template <typename Close_Function>
  void close_descriptor(socket_type descriptor, Close_Function close_function)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // We need to interrupt the select if any operations were cancelled.
    bool interrupt = read_op_queue_.cancel_operations(descriptor);
    interrupt = write_op_queue_.cancel_operations(descriptor) || interrupt;
    interrupt = except_op_queue_.cancel_operations(descriptor) || interrupt;

    if (interrupt && select_in_progress_)
    {
      // The close function cannot be called on a descriptor while the select
      // call is running with that descriptor in an fd_set, so we schedule a
      // dummy timer to perform the socket close when the select has been
      // interrupted.
      void* token = 0;
      interrupt = timer_queue_.enqueue_timer(detail::time(0, 0),
          close_handler<Close_Function>(descriptor, close_function), token)
        || interrupt;
    }
    else
    {
      // Not currently using the descriptor in select so we can close it now.
      close_function(descriptor);
    }

    if (interrupt)
      interrupter_.interrupt();
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

    // Dispatch any operation cancellations that were made while the select
    // loop was not running.
    read_op_queue_.dispatch_cancellations();
    write_op_queue_.dispatch_cancellations();
    except_op_queue_.dispatch_cancellations();

    bool stop = false;
    while (!stop)
    {
      // Set up the descriptor sets.
      fd_set_adaptor read_fds;
      read_fds.set(interrupter_.read_descriptor());
      read_op_queue_.get_descriptors(read_fds);
      fd_set_adaptor write_fds;
      write_op_queue_.get_descriptors(write_fds);
      fd_set_adaptor except_fds;
      except_op_queue_.get_descriptors(except_fds);
      socket_type max_fd = read_fds.max_descriptor();
      if (write_fds.max_descriptor() > max_fd)
        max_fd = write_fds.max_descriptor();
      if (except_fds.max_descriptor() > max_fd)
        max_fd = except_fds.max_descriptor();

      // Block on the select call without holding the lock so that new
      // operations can be started while the call is executing.
      timeval tv_buf;
      timeval* tv = get_timeout(tv_buf);
      select_in_progress_ = true;
      lock.unlock();
      int retval = socket_ops::select(static_cast<int>(max_fd + 1),
          read_fds, write_fds, except_fds, tv);
      lock.lock();
      select_in_progress_ = false;

      // Reset the interrupter.
      if (read_fds.is_set(interrupter_.read_descriptor()))
        stop = interrupter_.reset();

      // Dispatch all ready operations.
      if (retval > 0)
      {
        read_op_queue_.dispatch_descriptors(read_fds);
        write_op_queue_.dispatch_descriptors(write_fds);
        except_op_queue_.dispatch_descriptors(except_fds);
        read_op_queue_.dispatch_cancellations();
        write_op_queue_.dispatch_cancellations();
        except_op_queue_.dispatch_cancellations();
      }
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

    detail::time now = detail::time::now();
    detail::time earliest_timer;
    timer_queue_.get_earliest_time(earliest_timer);
    if (now < earliest_timer)
    {
      detail::time timeout = earliest_timer;
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

  // Cancel all operations associated with the given descriptor. The do_cancel
  // function of the handler objects will be invoked. This function does not
  // acquire the select_reactor's mutex.
  void cancel_ops_unlocked(socket_type descriptor)
  {
    bool interrupt = read_op_queue_.cancel_operations(descriptor);
    interrupt = write_op_queue_.cancel_operations(descriptor) || interrupt;
    interrupt = except_op_queue_.cancel_operations(descriptor) || interrupt;
    if (interrupt)
      interrupter_.interrupt();
  }

  // Adapts the FD_SET type to meet the Descriptor_Set concept's requirements.
  class fd_set_adaptor
  {
  public:
    fd_set_adaptor()
      : max_descriptor_(invalid_socket)
    {
      FD_ZERO(&fd_set_);
    }

    void set(socket_type descriptor)
    {
      if (max_descriptor_ == invalid_socket || descriptor > max_descriptor_)
        max_descriptor_ = descriptor;
      FD_SET(descriptor, &fd_set_);
    }

    bool is_set(socket_type descriptor) const
    {
      return FD_ISSET(descriptor, &fd_set_) != 0;
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

  // Whether the select loop is currently running or not.
  bool select_in_progress_;

  // The interrupter is used to break a blocking select call.
  select_interrupter interrupter_;

  // The queue of read operations.
  reactor_op_queue<socket_type> read_op_queue_;

  // The queue of write operations.
  reactor_op_queue<socket_type> write_op_queue_;

  // The queue of exception operations.
  reactor_op_queue<socket_type> except_op_queue_;

  // The queue of timers.
  reactor_timer_queue<detail::time> timer_queue_;

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

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SELECT_REACTOR_HPP
