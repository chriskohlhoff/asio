//
// dgram_socket.hpp
// ~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DGRAM_SOCKET_HPP
#define ASIO_DGRAM_SOCKET_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_dgram_socket.hpp"
#include "asio/demuxer.hpp"
#if defined(_WIN32)
# include "asio/detail/win_iocp_dgram_socket_service.hpp"
#else
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
