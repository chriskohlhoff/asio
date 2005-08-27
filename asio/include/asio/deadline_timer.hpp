//
// deadline_timer.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DEADLINE_TIMER_HPP
#define ASIO_DEADLINE_TIMER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_deadline_timer.hpp"
#include "asio/deadline_timer_service.hpp"

namespace asio {

/// Typedef for the typical usage of timer.
typedef basic_deadline_timer<deadline_timer_service<> > deadline_timer;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DEADLINE_TIMER_HPP
