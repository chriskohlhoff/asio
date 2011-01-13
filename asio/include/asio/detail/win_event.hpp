//
// detail/win_event.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_EVENT_HPP
#define ASIO_DETAIL_WIN_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(BOOST_WINDOWS)

#include <boost/assert.hpp>
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class win_event
  : private noncopyable
{
public:
  // Constructor.
  ASIO_DECL win_event();

  // Destructor.
  ~win_event()
  {
    ::CloseHandle(event_);
  }

  // Signal the event.
  template <typename Lock>
  void signal(Lock& lock)
  {
    BOOST_ASSERT(lock.locked());
    (void)lock;
    ::SetEvent(event_);
  }

  // Signal the event and unlock the mutex.
  template <typename Lock>
  void signal_and_unlock(Lock& lock)
  {
    BOOST_ASSERT(lock.locked());
    lock.unlock();
    ::SetEvent(event_);
  }

  // Reset the event.
  template <typename Lock>
  void clear(Lock& lock)
  {
    BOOST_ASSERT(lock.locked());
    (void)lock;
    ::ResetEvent(event_);
  }

  // Wait for the event to become signalled.
  template <typename Lock>
  void wait(Lock& lock)
  {
    BOOST_ASSERT(lock.locked());
    lock.unlock();
    ::WaitForSingleObject(event_, INFINITE);
    lock.lock();
  }

private:
  HANDLE event_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_event.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(BOOST_WINDOWS)

#endif // ASIO_DETAIL_WIN_EVENT_HPP
