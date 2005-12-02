//
// win_event.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_EVENT_HPP
#define ASIO_DETAIL_WIN_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#if defined(BOOST_WINDOWS)

#include "asio/system_exception.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

class win_event
  : private noncopyable
{
public:
  // Constructor.
  win_event()
    : event_(::CreateEvent(0, true, false, 0))
  {
    if (!event_)
    {
      DWORD last_error = ::GetLastError();
      system_exception e("event", last_error);
      boost::throw_exception(e);
    }
  }

  // Destructor.
  ~win_event()
  {
    ::CloseHandle(event_);
  }

  // Signal the event.
  void signal()
  {
    ::SetEvent(event_);
  }

  // Reset the event.
  void clear()
  {
    ::ResetEvent(event_);
  }

  // Wait for the event to become signalled.
  void wait()
  {
    ::WaitForSingleObject(event_, INFINITE);
  }

private:
  HANDLE event_;
};

} // namespace detail
} // namespace asio

#endif // defined(BOOST_WINDOWS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_EVENT_HPP
