//
// timer.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_TIMER_HPP
#define ASIO_TIMER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_timer.hpp"
#include "asio/demuxer.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_timer_service.hpp"

namespace asio {

/// Typedef for the typical usage of timer.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_timer
  <
    implementation_defined
  > timer;
#elif defined(ASIO_HAS_EPOLL_REACTOR)
typedef basic_timer
  <
    detail::reactive_timer_service
      <
        demuxer,
        detail::epoll_reactor
      >
  > timer;
#else
typedef basic_timer
  <
    detail::reactive_timer_service
      <
        demuxer,
        detail::select_reactor
      >
  > timer;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TIMER_HPP
