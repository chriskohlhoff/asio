//
// dgram_socket.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DGRAM_SOCKET_HPP
#define ASIO_DGRAM_SOCKET_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_dgram_socket.hpp"
#include "asio/demuxer.hpp"
#if defined(_WIN32)
# include "asio/detail/win_iocp_dgram_socket_service.hpp"
#else
# include "asio/detail/epoll_reactor.hpp"
# include "asio/detail/select_reactor.hpp"
# include "asio/detail/reactive_dgram_socket_service.hpp"
#endif

namespace asio {

/// Typedef for the typical usage of dgram_socket.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_dgram_socket
  <
    implementation_defined
  > dgram_socket;
#elif defined(_WIN32)
typedef basic_dgram_socket
  <
    detail::win_iocp_dgram_socket_service
  > dgram_socket;
#elif (ASIO_HAS_EPOLL_REACTOR)
typedef basic_dgram_socket
  <
    detail::reactive_dgram_socket_service
      <
        demuxer,
        detail::epoll_reactor
      >
  > dgram_socket;
#else
typedef basic_dgram_socket
  <
    detail::reactive_dgram_socket_service
      <
        demuxer,
        detail::select_reactor
      >
  > dgram_socket;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DGRAM_SOCKET_HPP
