//
// stream_socket.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_STREAM_SOCKET_HPP
#define ASIO_STREAM_SOCKET_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_stream_socket.hpp"
#include "asio/demuxer.hpp"
#if defined(_WIN32)
# include "asio/detail/win_iocp_stream_socket_service.hpp"
#else
# include "asio/detail/epoll_reactor.hpp"
# include "asio/detail/select_reactor.hpp"
# include "asio/detail/reactive_stream_socket_service.hpp"
#endif

namespace asio {

/// Typedef for the typical usage of stream_socket.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_stream_socket
  <
    implementation_defined
  > stream_socket;
#elif defined(_WIN32)
typedef basic_stream_socket
  <
    detail::win_iocp_stream_socket_service
  > stream_socket;
#elif defined(ASIO_HAS_EPOLL_REACTOR)
typedef basic_stream_socket
  <
    detail::reactive_stream_socket_service
      <
        demuxer,
        detail::epoll_reactor
      >
  > stream_socket;
#else
typedef basic_stream_socket
  <
    detail::reactive_stream_socket_service
      <
        demuxer,
        detail::select_reactor
      >
  > stream_socket;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_STREAM_SOCKET_HPP
