//
// socket_acceptor.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SOCKET_ACCEPTOR_HPP
#define ASIO_SOCKET_ACCEPTOR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_socket_acceptor.hpp"
#include "asio/demuxer.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_socket_acceptor_service.hpp"

namespace asio {

/// Typedef for the typical usage of socket_acceptor.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_socket_acceptor
  <
    implementation_defined
  > socket_acceptor;
#elif defined(ASIO_HAS_EPOLL_REACTOR)
typedef basic_socket_acceptor
  <
    detail::reactive_socket_acceptor_service
      <
        demuxer,
        detail::epoll_reactor
      >
  > socket_acceptor;
#else
typedef basic_socket_acceptor
  <
    detail::reactive_socket_acceptor_service
      <
        demuxer,
        detail::select_reactor
      >
  > socket_acceptor;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_ACCEPTOR_HPP
