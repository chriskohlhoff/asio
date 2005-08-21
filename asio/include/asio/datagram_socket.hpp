//
// datagram_socket.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DATAGRAM_SOCKET_HPP
#define ASIO_DATAGRAM_SOCKET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_datagram_socket.hpp"
#include "asio/demuxer.hpp"
#if defined(_WIN32)
# include "asio/detail/win_iocp_datagram_socket_service.hpp"
#else
# include "asio/detail/epoll_reactor.hpp"
# include "asio/detail/select_reactor.hpp"
# include "asio/detail/reactive_datagram_socket_service.hpp"
#endif

namespace asio {

/// Typedef for the typical usage of datagram_socket.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_datagram_socket
  <
    implementation_defined
  > datagram_socket;
#elif defined(_WIN32)
typedef basic_datagram_socket
  <
    detail::win_iocp_datagram_socket_service
  > datagram_socket;
#elif (ASIO_HAS_EPOLL_REACTOR)
typedef basic_datagram_socket
  <
    detail::reactive_datagram_socket_service
      <
        demuxer,
        detail::epoll_reactor
      >
  > datagram_socket;
#else
typedef basic_datagram_socket
  <
    detail::reactive_datagram_socket_service
      <
        demuxer,
        detail::select_reactor
      >
  > datagram_socket;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DATAGRAM_SOCKET_HPP
