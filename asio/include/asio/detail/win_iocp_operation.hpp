//
// win_iocp_operation.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_OPERATION_HPP
#define ASIO_DETAIL_WIN_IOCP_OPERATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

// This class is only supported on Win32 (NT4 and later).
#if defined(BOOST_WINDOWS)
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)

#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

class win_iocp_demuxer_service;

// Base class for all IOCP operations. A function pointer is used instead of
// virtual functions to avoid the associated overhead.
//
// This class inherits from OVERLAPPED so that we can downcast to get back to
// the win_iocp_operation pointer from the LPOVERLAPPED out parameter of
// GetQueuedCompletionStatus.
struct win_iocp_operation
  : public OVERLAPPED
{
  typedef void (*func_type)(win_iocp_operation*, DWORD, size_t);

  win_iocp_operation(func_type func)
    : func_(func)
  {
    Internal = 0;
    InternalHigh = 0;
    Offset = 0;
    OffsetHigh = 0;
    hEvent = 0;
  }

  void do_completion(DWORD last_error, size_t bytes_transferred)
  {
    func_(this, last_error, bytes_transferred);
  }

protected:
  // Prevent deletion through this type.
  ~win_iocp_operation()
  {
  }

private:
  func_type func_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)
#endif // defined(BOOST_WINDOWS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_OPERATION_HPP
