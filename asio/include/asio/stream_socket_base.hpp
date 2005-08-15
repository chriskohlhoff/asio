//
// stream_socket_base.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_STREAM_SOCKET_BASE_HPP
#define ASIO_STREAM_SOCKET_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

namespace asio {

/// The stream_socket_base class is used as a base for the basic_stream_socket
/// class template so that we have a common place to define the shutdown_type
/// enum.
class stream_socket_base
{
public:
  /// Different ways a socket may be shutdown.
  enum shutdown_type
  {
    /// Shutdown the receive side of the socket.
    shutdown_recv,

    /// Shutdown the send side of the socket.
    shutdown_send,

    /// Shutdown both send and receive on the socket.
    shutdown_both
  };

protected:
  /// Protected destructor to prevent deletion through this type.
  ~stream_socket_base()
  {
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_STREAM_SOCKET_BASE_HPP
