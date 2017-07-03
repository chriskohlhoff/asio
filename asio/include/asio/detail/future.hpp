//
// detail/array.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2016 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_FUTURE_HPP
#define ASIO_DETAIL_FUTURE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_FUTURE)
# include <future>
#else // defined(ASIO_HAS_STD_FUTURE)
# define ASIO_USING_BOOST_FUTURE
# include <boost/thread/future.hpp>
#endif // defined(ASIO_HAS_STD_FUTURE)

namespace asio {
namespace detail {

#if defined(ASIO_HAS_STD_FUTURE)
using std::future;
using std::packaged_task;
using std::promise;
#else // defined(ASIO_HAS_STD_FUTURE)
#if BOOST_THREAD_VERSION < 3
#	error BOOST_THREAD_VERSION must be defined to at least 3
#endif
using boost::future;
using boost::packaged_task;
using boost::promise;
#endif // defined(ASIO_HAS_STD_FUTURE)

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_FUTURE_HPP
