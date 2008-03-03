//
// mutex.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_MUTEX_HPP
#define ASIO_DETAIL_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#if !defined(BOOST_HAS_THREADS)
# include "asio/detail/null_mutex.hpp"
#elif defined(BOOST_WINDOWS)
# include "asio/detail/win_mutex.hpp"
#elif defined(BOOST_HAS_PTHREADS)
# include "asio/detail/posix_mutex.hpp"
#else
# error Only Windows and POSIX are supported!
#endif

namespace asio {
namespace detail {

#if !defined(BOOST_HAS_THREADS)
typedef null_mutex mutex;
#elif defined(BOOST_WINDOWS)
typedef win_mutex mutex;
#elif defined(BOOST_HAS_PTHREADS)
typedef posix_mutex mutex;
#endif

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_MUTEX_HPP
