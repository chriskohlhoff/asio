//
// locking_dispatcher.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_LOCKING_DISPATCHER_HPP
#define ASIO_LOCKING_DISPATCHER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_locking_dispatcher.hpp"
#include "asio/demuxer.hpp"
#include "asio/detail/locking_dispatcher_service.hpp"

namespace asio {

/// Typedef for the typical usage of locking_dispatcher.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_locking_dispatcher
  <
    implementation_defined
  > locking_dispatcher;
#else
typedef basic_locking_dispatcher
  <
    detail::locking_dispatcher_service<demuxer>
  > locking_dispatcher;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_LOCKING_DISPATCHER_HPP
