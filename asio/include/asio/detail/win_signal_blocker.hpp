//
// win_signal_blocker.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_SIGNAL_BLOCKER_HPP
#define ASIO_DETAIL_WIN_SIGNAL_BLOCKER_HPP

#include "asio/detail/push_options.hpp"

#if defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

class win_signal_blocker
  : private boost::noncopyable
{
public:
  // Constructor blocks all signals for the calling thread.
  win_signal_blocker()
  {
    // No-op.
  }

  // Destructor restores the previous signal mask.
  ~win_signal_blocker()
  {
    // No-op.
  }

  // Block all signals for the calling thread.
  void block()
  {
    // No-op.
  }

  // Restore the previous signal mask.
  void unblock()
  {
    // No-op.
  }
};

} // namespace detail
} // namespace asio

#endif // _WIN32

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_SIGNAL_BLOCKER_HPP
