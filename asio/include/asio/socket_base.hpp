//
// socket_base.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SOCKET_BASE_HPP
#define ASIO_SOCKET_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/socket_types.hpp"

namespace asio {

/// The socket_base class is used as a base for the basic_stream_socket and
/// basic_datagram_socket class templates so that we have a common place to
/// define the shutdown_type and enum.
class socket_base
{
public:
  /// Different ways a socket may be shutdown.
  enum shutdown_type
  {
#if defined(GENERATING_DOCUMENTATION)
    /// Shutdown the receive side of the socket.
    shutdown_receive = implementation_defined,

    /// Shutdown the send side of the socket.
    shutdown_send = implementation_defined,

    /// Shutdown both send and receive on the socket.
    shutdown_both = implementation_defined
#else
    shutdown_receive = asio::detail::shutdown_receive,
    shutdown_send = asio::detail::shutdown_send,
    shutdown_both = asio::detail::shutdown_both
#endif
  };

  /// Flags that can be passed to send and receive operations.
  enum message_flags
  {
#if defined(GENERATING_DOCUMENTATION)
    /// Peek at incoming data without removing it from the input queue.
    message_peek = implementation_defined,

    /// Process out-of-band data.
    message_out_of_band = implementation_defined,

    /// Specify that the data should not be subject to routing.
    message_do_not_route = implementation_defined
#else
    message_peek = asio::detail::message_peek,
    message_out_of_band = asio::detail::message_out_of_band,
    message_do_not_route = asio::detail::message_do_not_route
#endif
  };

protected:
  /// Protected destructor to prevent deletion through this type.
  ~socket_base()
  {
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_BASE_HPP
