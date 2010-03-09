//
// reactor_fwd.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTOR_FWD_HPP
#define ASIO_DETAIL_REACTOR_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/dev_poll_reactor_fwd.hpp"
#include "asio/detail/epoll_reactor_fwd.hpp"
#include "asio/detail/kqueue_reactor_fwd.hpp"
#include "asio/detail/select_reactor_fwd.hpp"
#include "asio/detail/win_iocp_io_service_fwd.hpp"

namespace asio {
namespace detail {

#if defined(ASIO_HAS_IOCP)
typedef select_reactor<true> reactor;
#elif defined(ASIO_HAS_EPOLL)
typedef epoll_reactor reactor;
#elif defined(ASIO_HAS_KQUEUE)
typedef kqueue_reactor reactor;
#elif defined(ASIO_HAS_DEV_POLL)
typedef dev_poll_reactor reactor;
#else
typedef select_reactor<false> reactor;
#endif

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTOR_FWD_HPP
