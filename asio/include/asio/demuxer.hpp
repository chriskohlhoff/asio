//
// demuxer.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DEMUXER_HPP
#define ASIO_DEMUXER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/service_factory.hpp"
#if defined(_WIN32)
# include "asio/detail/win_iocp_demuxer_service.hpp"
#else
# include "asio/detail/epoll_reactor.hpp"
# include "asio/detail/select_reactor.hpp"
# include "asio/detail/task_demuxer_service.hpp"
#endif

namespace asio {

/// Typedef for typical usage of demuxer.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_demuxer
  <
    implementation_defined
  > demuxer;
#elif defined(_WIN32)
typedef basic_demuxer
  <
    detail::win_iocp_demuxer_service
  > demuxer;
#elif defined(ASIO_HAS_EPOLL_REACTOR)
typedef basic_demuxer
  <
    detail::task_demuxer_service
      <
        detail::epoll_reactor
      >
  > demuxer;
#else
typedef basic_demuxer
  <
    detail::task_demuxer_service
      <
        detail::select_reactor
      >
  > demuxer;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DEMUXER_HPP
