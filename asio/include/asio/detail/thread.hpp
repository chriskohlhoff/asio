//
// thread.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_THREAD_HPP
#define ASIO_DETAIL_THREAD_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/posix_thread.hpp"
#include "asio/detail/win_thread.hpp"

namespace asio {
namespace detail {

#if defined(_WIN32)
typedef win_thread thread;
#else
typedef posix_thread thread;
#endif

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_THREAD_HPP
