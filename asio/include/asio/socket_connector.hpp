//
// socket_connector.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SOCKET_CONNECTOR_HPP
#define ASIO_SOCKET_CONNECTOR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_socket_connector.hpp"
#include "asio/demuxer.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_socket_connector_service.hpp"

namespace asio {

/// Typedef for the typical usage of socket_connector.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_socket_connector
  <
    implementation_defined
  > socket_connector;
#else
typedef basic_socket_connector
  <
    detail::reactive_socket_connector_service
      <
        demuxer,
        detail::select_reactor
      >
  > socket_connector;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_CONNECTOR_HPP
