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
# include <exception>
#else // defined(ASIO_HAS_STD_FUTURE)
# if defined(BOOST_THREAD_VERSION) && (BOOST_THREAD_VERSION < 3)
#   error BOOST_THREAD_VERSION must be defined to at least 3
# else
#   if !defined(BOOST_THREAD_VERSION)
#     define BOOST_THREAD_VERSION 3
#   endif
# endif
# define ASIO_USING_BOOST_FUTURE
# include <boost/thread/future.hpp>
#endif // defined(ASIO_HAS_STD_FUTURE)

namespace asio {
namespace detail {

#if defined(ASIO_HAS_STD_FUTURE)
using std::future;
using std::packaged_task;
using std::promise;
#define ASIO_CURRENT_EXCEPTION std::current_exception()
#define ASIO_MAKE_EXCEPTION_PTR(_ex) std::make_exception_ptr(_ex)
using std::exception_ptr;
#else // defined(ASIO_HAS_STD_FUTURE)
using boost::future;
using boost::packaged_task;
using boost::promise;
#define ASIO_CURRENT_EXCEPTION boost::current_exception()
#define ASIO_MAKE_EXCEPTION_PTR(_ex) boost::copy_exception(_ex)
using boost::exception_ptr;
#endif // defined(ASIO_HAS_STD_FUTURE)
} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_FUTURE_HPP
