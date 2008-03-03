//
// kqueue_reactor_fwd.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2005 Stefan Arentz (stefan at soze dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_KQUEUE_REACTOR_FWD_HPP
#define ASIO_DETAIL_KQUEUE_REACTOR_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#if !defined(ASIO_DISABLE_KQUEUE)
#if defined(__MACH__) && defined(__APPLE__)

// Define this to indicate that epoll is supported on the target platform.
#define ASIO_HAS_KQUEUE 1

namespace asio {
namespace detail {

template <bool Own_Thread>
class kqueue_reactor;

} // namespace detail
} // namespace asio

#endif // defined(__MACH__) && defined(__APPLE__)
#endif // !defined(ASIO_DISABLE_KQUEUE)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_KQUEUE_REACTOR_FWD_HPP
