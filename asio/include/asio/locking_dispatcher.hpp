//
// locking_dispatcher.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
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
