//
// null_thread.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_NULL_THREAD_HPP
#define ASIO_DETAIL_NULL_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#if !defined(BOOST_HAS_THREADS)

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/noncopyable.hpp"

namespace asio {
namespace detail {

class null_thread
  : private noncopyable
{
public:
  // Constructor.
  template <typename Function>
  null_thread(Function f)
  {
    asio::system_error e(
        asio::error::operation_not_supported, "thread");
    boost::throw_exception(e);
  }

  // Destructor.
  ~null_thread()
  {
  }

  // Wait for the thread to exit.
  void join()
  {
  }
};

} // namespace detail
} // namespace asio

#endif // !defined(BOOST_HAS_THREADS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_NULL_THREAD_HPP
