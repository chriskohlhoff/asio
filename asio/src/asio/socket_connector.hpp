//
// socket_connector.hpp
// ~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_SOCKET_CONNECTOR_HPP
#define ASIO_SOCKET_CONNECTOR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_socket_connector.hpp"
#include "asio/demuxer.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_socket_connector_service.hpp"

namespace asio {

/// Typedef for the typical usage of socket_connector.
typedef basic_socket_connector
  <
    detail::reactive_socket_connector_service
      <
        demuxer,
        detail::select_reactor<demuxer>
      >
  > socket_connector;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_CONNECTOR_HPP
