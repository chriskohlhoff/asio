//
// timer.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_TIMER_HPP
#define ASIO_TIMER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_timer.hpp"
#include "asio/demuxer.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_timer_service.hpp"

namespace asio {

/// Typedef for the typical usage of timer.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_timer
  <
    implementation_defined
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
