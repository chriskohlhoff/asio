//
// stream_socket_base.hpp
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

#ifndef ASIO_STREAM_SOCKET_BASE_HPP
#define ASIO_STREAM_SOCKET_BASE_HPP

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
