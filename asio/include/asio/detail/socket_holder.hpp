//
// socket_holder.hpp
// ~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SOCKET_HOLDER_HPP
#define ASIO_DETAIL_SOCKET_HOLDER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_ops.hpp"

namespace asio {
namespace detail {

// Implement the resource acquisition is initialisation idiom for sockets.
class socket_holder
  : private boost::noncopyable
{
public:
  // Construct as an uninitialised socket.
  socket_holder()
    : socket_(invalid_socket)
  {
  }

  // Construct to take ownership of the specified socket.
  explicit socket_holder(socket_type s)
    : socket_(s)
  {
  }

  // Destructor.
  ~socket_holder()
  {
    if (socket_ != invalid_socket)
      socket_ops::close(socket_);
  }

  // Get the underlying socket.
  socket_type get() const
  {
    return socket_;
  }

  // Reset to an uninitialised socket.
  void reset()
  {
    if (socket_ != invalid_socket)
    {
      socket_ops::close(socket_);
      socket_ = invalid_socket;
    }
  }

  // Reset to take ownership of the specified socket.
  void reset(socket_type s)
  {
    reset();
    socket_ = s;
  }

  // Release ownership of the socket.
  socket_type release()
  {
    socket_type tmp = socket_;
    socket_ = invalid_socket;
    return tmp;
  }

private:
  // The underlying socket.
  socket_type socket_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_HOLDER_HPP
