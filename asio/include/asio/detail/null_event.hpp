//
// null_event.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_NULL_EVENT_HPP
#define ASIO_DETAIL_NULL_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#if !defined(BOOST_HAS_THREADS)

#include "asio/detail/noncopyable.hpp"

namespace asio {
namespace detail {

class null_event
  : private noncopyable
{
public:
  // Constructor.
  null_event()
  {
  }

  // Destructor.
  ~null_event()
  {
  }

  // Signal the event.
  void signal()
  {
  }

  // Reset the event.
  void clear()
  {
  }

  // Wait for the event to become signalled.
  void wait()
  {
  }
};

} // namespace detail
} // namespace asio

#endif // !defined(BOOST_HAS_THREADS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_NULL_EVENT_HPP
