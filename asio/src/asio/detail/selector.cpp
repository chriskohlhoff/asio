//
// selector.cpp
// ~~~~~~~~~~~~
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

#include "asio/detail/selector.hpp"
#include <csignal>
#include <boost/bind.hpp>
#include "asio/demuxer.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

selector::
selector(
    demuxer& d)
  : mutex_(),
    demuxer_(d),
    interrupter_(),
    read_op_queue_(),
    write_op_queue_()
{
#if !defined(_WIN32)
  std::signal(SIGPIPE, SIG_IGN);
#endif // !defined(_WIN32)

  demuxer_.add_task(*this, 0);
}

selector::
~selector()
{
}

void
selector::
start_read_op(
    select_op& op)
{
  boost::mutex::scoped_lock lock(mutex_);

  if (read_op_queue_.enqueue_operation(op))
    interrupter_.interrupt();
}

void
selector::
restart_read_op(
    select_op& op)
{
  read_op_queue_.enqueue_operation(op);
}

void
selector::
start_write_op(
    select_op& op)
{
  boost::mutex::scoped_lock lock(mutex_);

  if (write_op_queue_.enqueue_operation(op))
    interrupter_.interrupt();
}

void
selector::
restart_write_op(
    select_op& op)
{
  write_op_queue_.enqueue_operation(op);
}

void
selector::
close_descriptor(
    socket_type descriptor)
{
  boost::mutex::scoped_lock lock(mutex_);

  read_op_queue_.close_descriptor(descriptor);
  write_op_queue_.close_descriptor(descriptor);
}

void
selector::
prepare_task(
    void*)
  throw ()
{
  try
  {
    interrupter_.reset();
  }
  catch (...)
  {
  }
}

bool
selector::
execute_task(
    const boost::xtime& interval,
    void*)
  throw ()
{
  try
  {
    boost::mutex::scoped_lock lock(mutex_);

    while (!interrupter_.reset())
    {
      // Set up the read descriptor set.
      fd_set read_fds;
      FD_ZERO(&read_fds);
      FD_SET(interrupter_.read_descriptor(), &read_fds);
      int max_fd = interrupter_.read_descriptor();
      int max_read_fd = read_op_queue_.get_descriptors(read_fds);
      if (max_read_fd > max_fd)
        max_fd = max_read_fd;

      // Set up the write descriptor set.
      fd_set write_fds;
      FD_ZERO(&write_fds);
      int max_write_fd = write_op_queue_.get_descriptors(write_fds);
      if (max_write_fd > max_fd)
        max_fd = max_write_fd;

      // Block on the select call without holding the lock so that new
      // operations can be started while the call is executing. TODO pass the
      // interval.
      lock.unlock();
      int retval = ::select(max_fd + 1, &read_fds, &write_fds, 0, 0);
      lock.lock();

      // Dispatch all ready operations.
      if (retval > 0)
      {
        read_op_queue_.dispatch_descriptors(read_fds);
        write_op_queue_.dispatch_descriptors(write_fds);
      }
    }

    return false;
  }
  catch (...)
  {
    return true;
  }
}

void
selector::
interrupt_task(
    void*)
  throw ()
{
  try
  {
    interrupter_.interrupt();
  }
  catch (...)
  {
  }
}

} // namespace detail
} // namespace asio
