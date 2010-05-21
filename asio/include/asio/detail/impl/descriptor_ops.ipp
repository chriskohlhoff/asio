//
// detail/impl/descriptor_ops.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_DESCRIPTOR_OPS_IPP
#define ASIO_DETAIL_IMPL_DESCRIPTOR_OPS_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <cerrno>
#include "asio/detail/descriptor_ops.hpp"
#include "asio/error.hpp"

#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {
namespace descriptor_ops {

template <typename ReturnType>
inline ReturnType error_wrapper(ReturnType return_value,
    asio::error_code& ec)
{
  ec = asio::error_code(errno,
      asio::error::get_system_category());
  return return_value;
}

int open(const char* path, int flags, asio::error_code& ec)
{
  errno = 0;
  int result = error_wrapper(::open(path, flags), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
}

int close(int d, asio::error_code& ec)
{
  errno = 0;
  int result = (d != -1) ? error_wrapper(::close(d), ec) : 0;
  if (result == 0)
    ec = asio::error_code();
  return result;
}

std::size_t sync_read(int d, buf* bufs, std::size_t count,
    bool all_empty, bool non_blocking, asio::error_code& ec)
{
  if (d == -1)
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  // A request to read 0 bytes on a stream is a no-op.
  if (all_empty)
  {
    ec = asio::error_code();
    return 0;
  }

  // Read some data.
  for (;;)
  {
    // Try to complete the operation without blocking.
    errno = 0;
    int bytes = error_wrapper(::readv(d, bufs, static_cast<int>(count)), ec);

    // Check if operation succeeded.
    if (bytes > 0)
      return bytes;

    // Check for EOF.
    if (bytes == 0)
    {
      ec = asio::error::eof;
      return 0;
    }

    // Operation failed.
    if (non_blocking
        || (ec != asio::error::would_block
          && ec != asio::error::try_again))
      return 0;

    // Wait for descriptor to become ready.
    if (descriptor_ops::poll_read(d, ec) < 0)
      return 0;
  }
}

bool non_blocking_read(int d, buf* bufs, size_t count,
    asio::error_code& ec, std::size_t& bytes_transferred)
{
  for (;;)
  {
    // Read some data.
    errno = 0;
    int bytes = error_wrapper(::readv(d, bufs, static_cast<int>(count)), ec);

    // Check for end of stream.
    if (bytes == 0)
    {
      ec = asio::error::eof;
      return 0;
    }

    // Retry operation if interrupted by signal.
    if (ec == asio::error::interrupted)
      continue;

    // Check if we need to run the operation again.
    if (ec == asio::error::would_block
        || ec == asio::error::try_again)
      return false;

    // Operation is complete.
    if (bytes >= 0)
    {
      ec = asio::error_code();
      bytes_transferred = bytes;
    }
    else
      bytes_transferred = 0;

    return true;
  }
}

std::size_t sync_write(int d, const buf* bufs, std::size_t count,
    bool all_empty, bool non_blocking, asio::error_code& ec)
{
  if (d == -1)
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  // A request to write 0 bytes on a stream is a no-op.
  if (all_empty)
  {
    ec = asio::error_code();
    return 0;
  }

  // Write some data.
  for (;;)
  {
    // Try to complete the operation without blocking.
    errno = 0;
    int bytes = error_wrapper(::writev(d, bufs, static_cast<int>(count)), ec);

    // Check if operation succeeded.
    if (bytes > 0)
      return bytes;

    // Operation failed.
    if (non_blocking
        || (ec != asio::error::would_block
          && ec != asio::error::try_again))
      return 0;

    // Wait for descriptor to become ready.
    if (descriptor_ops::poll_write(d, ec) < 0)
      return 0;
  }
}

bool non_blocking_write(int d, const buf* bufs, size_t count,
    asio::error_code& ec, std::size_t& bytes_transferred)
{
  for (;;)
  {
    // Write some data.
    errno = 0;
    int bytes = error_wrapper(::writev(d, bufs, static_cast<int>(count)), ec);

    // Retry operation if interrupted by signal.
    if (ec == asio::error::interrupted)
      continue;

    // Check if we need to run the operation again.
    if (ec == asio::error::would_block
        || ec == asio::error::try_again)
      return false;

    // Operation is complete.
    if (bytes >= 0)
    {
      ec = asio::error_code();
      bytes_transferred = bytes;
    }
    else
      bytes_transferred = 0;

    return true;
  }
}

int ioctl(int d, long cmd, ioctl_arg_type* arg,
    asio::error_code& ec)
{
  if (d == -1)
  {
    ec = asio::error::bad_descriptor;
    return -1;
  }

  errno = 0;
  int result = error_wrapper(::ioctl(d, cmd, arg), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
}

int fcntl(int d, long cmd, asio::error_code& ec)
{
  if (d == -1)
  {
    ec = asio::error::bad_descriptor;
    return -1;
  }

  errno = 0;
  int result = error_wrapper(::fcntl(d, cmd), ec);
  if (result != -1)
    ec = asio::error_code();
  return result;
}

int fcntl(int d, long cmd, long arg, asio::error_code& ec)
{
  if (d == -1)
  {
    ec = asio::error::bad_descriptor;
    return -1;
  }

  errno = 0;
  int result = error_wrapper(::fcntl(d, cmd, arg), ec);
  if (result != -1)
    ec = asio::error_code();
  return result;
}

int poll_read(int d, asio::error_code& ec)
{
  if (d == -1)
  {
    ec = asio::error::bad_descriptor;
    return -1;
  }

  pollfd fds;
  fds.fd = d;
  fds.events = POLLIN;
  fds.revents = 0;
  errno = 0;
  int result = error_wrapper(::poll(&fds, 1, -1), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
}

int poll_write(int d, asio::error_code& ec)
{
  if (d == -1)
  {
    ec = asio::error::bad_descriptor;
    return -1;
  }

  pollfd fds;
  fds.fd = d;
  fds.events = POLLOUT;
  fds.revents = 0;
  errno = 0;
  int result = error_wrapper(::poll(&fds, 1, -1), ec);
  if (result >= 0)
    ec = asio::error_code();
  return result;
}

} // namespace descriptor_ops
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#endif // ASIO_DETAIL_IMPL_DESCRIPTOR_OPS_IPP
