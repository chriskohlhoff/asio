//
// dgram_socket.hpp
// ~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DGRAM_SOCKET_HPP
#define ASIO_DGRAM_SOCKET_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/function.hpp>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/completion_context.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {

class demuxer;
class dgram_socket_service;
class socket_address;
class socket_error;

/// The dgram_socket class provides asynchronous and blocking datagram-oriented
/// socket functionality.
class dgram_socket
  : private boost::noncopyable
{
public:
  /// The native type of the socket acceptor. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef detail::socket_type native_type;

  /// Construct a dgram_socket without opening it. The socket needs to be
  /// opened before data can be sent or received on it.
  explicit dgram_socket(demuxer& d);

  /// Construct a dgram_socket opened on the given address.
  dgram_socket(demuxer& d, const socket_address& address);

  /// Destructor.
  ~dgram_socket();

  /// Open the socket on the given address.
  void open(const socket_address& address);

  /// Close the socket.
  void close();

  /// Get the underlying handle in the native type.
  native_type native_handle() const;

  /// Send a datagram to the specified address. Returns the number of bytes
  /// sent. Throws a socket_error exception on failure.
  size_t sendto(const void* data, size_t length,
      const socket_address& destination);

  /// The handler when a sendto operation is completed. The first argument is
  /// the error code, the second is the number of bytes sent.
  typedef boost::function2<void, const socket_error&, size_t> sendto_handler;

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  void async_sendto(const void* data, size_t length,
      const socket_address& destination, const sendto_handler& handler,
      completion_context& context = completion_context::null());

  /// Receive a datagram with the address of the sender. Returns the number of
  /// bytes received. Throws a socket_error exception on failure.
  size_t recvfrom(void* data, size_t max_length,
      socket_address& sender_address);
  
  /// The handler when a recvfrom operation is completed. The first argument is
  /// the error code, the second is the number of bytes received.
  typedef boost::function2<void, const socket_error&, size_t> recvfrom_handler;

  /// Start an asynchronous receive. The buffer for the data being received and
  /// the sender_address obejct must both be valid for the lifetime of the
  /// asynchronous operation.
  void async_recvfrom(void* data, size_t max_length,
      socket_address& sender_address, const recvfrom_handler& handler,
      completion_context& context = completion_context::null());

private:
  /// The backend service implementation.
  dgram_socket_service& service_;

  /// The underlying native handle.
  native_type handle_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DGRAM_SOCKET_HPP
