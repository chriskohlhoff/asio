//
// eventfd_select_interrupter.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2008 Roelof Naude (roelof.naude at gmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_EVENTFD_SELECT_INTERRUPTER_HPP
#define ASIO_DETAIL_EVENTFD_SELECT_INTERRUPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#if defined(linux)
# if !defined(ASIO_DISABLE_EVENTFD)
#  include <linux/version.h>
#  if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,22)
#   define ASIO_HAS_EVENTFD
#  endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,22)
# endif // !defined(ASIO_DISABLE_EVENTFD)
#endif // defined(linux)

#if defined(ASIO_HAS_EVENTFD)

#include "asio/detail/push_options.hpp"
#include <fcntl.h>
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
# include <asm/unistd.h>
#else // __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
# include <sys/eventfd.h>
#endif // __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

class eventfd_select_interrupter
{
public:
  // Constructor.
  eventfd_select_interrupter()
  {
#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
    write_descriptor_ = read_descriptor_ = syscall(__NR_eventfd, 0);
#else // __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
    write_descriptor_ = read_descriptor_ = ::eventfd(0, 0);
#endif // __GLIBC__ == 2 && __GLIBC_MINOR__ < 8
    if (read_descriptor_ != -1)
    {
      ::fcntl(read_descriptor_, F_SETFL, O_NONBLOCK);
    }
    else
    {
      int pipe_fds[2];
      if (pipe(pipe_fds) == 0)
      {
        read_descriptor_ = pipe_fds[0];
        ::fcntl(read_descriptor_, F_SETFL, O_NONBLOCK);
        write_descriptor_ = pipe_fds[1];
        ::fcntl(write_descriptor_, F_SETFL, O_NONBLOCK);
      }
      else
      {
        asio::error_code ec(errno,
            asio::error::get_system_category());
        asio::system_error e(ec, "eventfd_select_interrupter");
        boost::throw_exception(e);
      }
    }
  }

  // Destructor.
  ~eventfd_select_interrupter()
  {
    if (write_descriptor_ != -1 && write_descriptor_ != read_descriptor_)
      ::close(write_descriptor_);
    if (read_descriptor_ != -1)
      ::close(read_descriptor_);
  }

  // Interrupt the select call.
  void interrupt()
  {
    uint64_t counter(1UL);
    int result = ::write(write_descriptor_, &counter, sizeof(uint64_t));
    (void)result;
  }

  // Reset the select interrupt. Returns true if the call was interrupted.
  bool reset()
  {
    if (write_descriptor_ == read_descriptor_)
    {
      // Only perform one read. The kernel maintains an atomic counter.
      uint64_t counter(0);
      int bytes_read = ::read(read_descriptor_, &counter, sizeof(uint64_t));
      bool was_interrupted = (bytes_read > 0);
      return was_interrupted;
    }
    else
    {
      // Clear all data from the pipe.
      char data[1024];
      int bytes_read = ::read(read_descriptor_, data, sizeof(data));
      bool was_interrupted = (bytes_read > 0);
      while (bytes_read == sizeof(data))
        bytes_read = ::read(read_descriptor_, data, sizeof(data));
      return was_interrupted;
    }
  }

  // Get the read descriptor to be passed to select.
  int read_descriptor() const
  {
    return read_descriptor_;
  }

private:
  // The read end of a connection used to interrupt the select call. This file
  // descriptor is passed to select such that when it is time to stop, a single
  // 64bit value will be written on the other end of the connection and this
  // descriptor will become readable.
  int read_descriptor_;

  // The write end of a connection used to interrupt the select call. A single
  // 64bit non-zero value may be written to this to wake up the select which is
  // waiting for the other end to become readable. This descriptor will only
  // differ from the read descriptor when a pipe is used.
  int write_descriptor_;
};

} // namespace detail
} // namespace asio

#endif // defined(ASIO_HAS_EVENTFD)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_EVENTFD_SELECT_INTERRUPTER_HPP
