//
// timer_queue.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_TIMER_QUEUE_HPP
#define ASIO_TIMER_QUEUE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_timer_queue.hpp"
#include "asio/detail/timer_queue_service.hpp"

namespace asio {

/// Typedef for the typical usage of timer_queue.
typedef basic_timer_queue<detail::timer_queue_service> timer_queue;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_TIMER_QUEUE_HPP
