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
#include <boost/thread.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/reactor_op_queue.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <typename Demuxer>
class select_reactor
  : private boost::noncopyable
{
public:
  // Constructor.
  select_reactor(Demuxer& d)
    : mutex_(),
      interrupter_(),
      read_op_queue_(),
      write_op_queue_(),
      stop_(false),
      thread_(boost::bind(&select_reactor<Demuxer>::run, this))
  {
  }

  // Destructor.
  ~select_reactor()
  {
    boost::mutex::scoped_lock lock(mutex_);
    stop_ = true;
    lock.unlock();
    interrupter_.interrupt();
    thread_.join();
  }

  // Start a new read operation. The do_operation function of the select_op
  // object will be invoked when the given descriptor is ready to be read.
  template <typename Handler>
  void start_read_op(socket_type descriptor, Handler handler)
  {
    boost::mutex::scoped_lock lock(mutex_);
    if (read_op_queue_.enqueue_operation(descriptor, handler))
      interrupter_.interrupt();
  }

  // Start a read operation from inside an op_call invocation. The do_operation
  // function of the select_op object will be invoked when the given descriptor
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
    boost::mutex::scoped_lock lock(mutex_);
    if (write_op_queue_.enqueue_operation(descriptor, handler))
      interrupter_.interrupt();
  }

  // Start a write operation from inside an op_call invocation. The
  // do_operation function of the select_op object will be invoked when the
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
    boost::mutex::scoped_lock lock(mutex_);
    read_op_queue_.close_descriptor(descriptor);
    write_op_queue_.close_descriptor(descriptor);
  }

private:
  // Run the select loop.
  void run()
  {
    boost::mutex::scoped_lock lock(mutex_);

    while (!stop_)
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
      // operations can be started while the call is executing. TODO pass the
      // interval.
      lock.unlock();
      int retval = ::select(max_fd + 1, read_fds, write_fds, 0, 0);
      lock.lock();

      // Reset the interrupter.
      if (read_fds.is_set(interrupter_.read_descriptor()))
        interrupter_.reset();

      // Dispatch all ready operations.
      if (retval > 0)
      {
        read_op_queue_.dispatch_descriptors(read_fds);
        write_op_queue_.dispatch_descriptors(write_fds);
      }
    }
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
  boost::mutex mutex_;

  // The interrupter is used to break a blocking select call.
  select_interrupter interrupter_;

  // The queue of read operations.
  reactor_op_queue<socket_type> read_op_queue_;

  // The queue of write operations.
  reactor_op_queue<socket_type> write_op_queue_;

  // Does the reactor loop need to stop.
  bool stop_;

  // The thread that is running the reactor loop.
  boost::thread thread_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SELECT_REACTOR_HPP
