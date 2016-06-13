//
// detail/std_mutex.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_STD_MUTEX_HPP
#define ASIO_DETAIL_STD_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "../detail/config.hpp"

#if defined(ASIO_HAS_STD_MUTEX_AND_CONDVAR)

#include <mutex>
#include "../detail/noncopyable.hpp"
#include "../detail/scoped_lock.hpp"

#include "../detail/push_options.hpp"

namespace asio {
namespace detail {

class std_event;

class std_mutex
  : private noncopyable
{
public:
  typedef asio::detail::scoped_lock<std_mutex> scoped_lock;

  // Constructor.
  std_mutex()
  {
  }

  // Destructor.
  ~std_mutex()
  {
  }

  // Lock the mutex.
  void lock()
  {
    mutex_.lock();
  }

  // Unlock the mutex.
  void unlock()
  {
    mutex_.unlock();
  }

private:
  friend class std_event;
  std::mutex mutex_;
};

} // namespace detail
} // namespace asio

#include "../detail/pop_options.hpp"

#endif // defined(ASIO_HAS_STD_MUTEX_AND_CONDVAR)

#endif // ASIO_DETAIL_STD_MUTEX_HPP
