//
// basic_dgram_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_BASIC_DGRAM_SOCKET_HPP
#define ASIO_BASIC_DGRAM_SOCKET_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/null_completion_context.hpp"
#include "asio/service_factory.hpp"

namespace asio {

/// The basic_dgram_socket class template provides asynchronous and blocking
/// datagram-oriented socket functionality. Most applications will simply use
/// the dgram_socket typedef.
template <typename Service>
class basic_dgram_socket
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide socket operations.
  typedef Service service_type;

  /// The native implementation type of the dgram socket.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// Construct a basic_dgram_socket without opening it. The socket needs to be
  /// opened before data can be sent or received on it.
  explicit basic_dgram_socket(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
  }

  /// Construct a basic_dgram_socket opened on the given address.
  template <typename Address>
  basic_dgram_socket(demuxer_type& d, const Address& address)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.create(impl_, address);
  }

  /// Destructor.
  ~basic_dgram_socket()
  {
    service_.destroy(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Open the socket on the given address.
  template <typename Address>
  void open(const Address& address)
  {
    service_.create(impl_, address);
  }

  /// Close the socket.
  void close()
  {
    service_.destroy(impl_);
  }

  /// Get the underlying implementation in the native type.
  impl_type impl()
  {
    return impl_;
  }

  /// Send a datagram to the specified address. Returns the number of bytes
  /// sent. Throws a socket_error exception on failure.
  template <typename Address>
  size_t sendto(const void* data, size_t length, const Address& destination)
  {
    return service_.sendto(impl_, data, length, destination);
  }

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Address, typename Handler>
  void async_sendto(const void* data, size_t length,
      const Address& destination, Handler handler)
  {
    service_.async_sendto(impl_, data, length, destination, handler,
        null_completion_context::instance());
  }

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Address, typename Handler, typename Completion_Context>
  void async_sendto(const void* data, size_t length,
      const Address& destination, Handler handler,
      Completion_Context& context)
  {
    service_.async_sendto(impl_, data, length, destination, handler, context);
  }

  /// Receive a datagram with the address of the sender. Returns the number of
  /// bytes received. Throws a socket_error exception on failure.
  template <typename Address>
  size_t recvfrom(void* data, size_t max_length, Address& sender_address)
  {
    return service_.recvfrom(impl_, data, max_length, sender_address);
  }
  
  /// Start an asynchronous receive. The buffer for the data being received and
  /// the sender_address obejct must both be valid for the lifetime of the
  /// asynchronous operation.
  template <typename Address, typename Handler>
  void async_recvfrom(void* data, size_t max_length, Address& sender_address,
      Handler handler)
  {
    service_.async_recvfrom(impl_, data, max_length, sender_address, handler,
        null_completion_context::instance());
  }

  /// Start an asynchronous receive. The buffer for the data being received and
  /// the sender_address obejct must both be valid for the lifetime of the
  /// asynchronous operation.
  template <typename Address, typename Handler, typename Completion_Context>
  void async_recvfrom(void* data, size_t max_length, Address& sender_address,
      Handler handler, Completion_Context& context)
  {
    service_.async_recvfrom(impl_, data, max_length, sender_address, handler,
        context);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_DGRAM_SOCKET_HPP
