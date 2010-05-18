//
// detail/impl/posix_thread.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_POSIX_THREAD_HPP
#define ASIO_DETAIL_IMPL_POSIX_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/posix_thread.hpp"

#if defined(BOOST_HAS_PTHREADS)

namespace asio {
namespace detail {

template <typename Function>
posix_thread::posix_thread(Function f)
  : joined_(false)
{
  start_thread(new func<Function>(f));
}

} // namespace detail
} // namespace asio

#endif // defined(BOOST_HAS_PTHREADS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_IMPL_POSIX_THREAD_HPP
