//
// basic_stream_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_BASIC_STREAM_SOCKET_HPP
#define ASIO_BASIC_STREAM_SOCKET_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/null_completion_context.hpp"
#include "asio/service_factory.hpp"

namespace asio {

/// The basic_stream_socket class template provides asynchronous and blocking
/// stream-oriented socket functionality. Most applications will simply use the
/// stream_socket typedef.
template <typename Service>
class basic_stream_socket
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide socket operations.
  typedef Service service_type;

  /// The native implementation type of the stream socket.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// A basic_stream_socket is always the lowest layer.
  typedef basic_stream_socket<service_type> lowest_layer_type;

  /// Construct a basic_stream_socket without opening it. The socket needs to
  /// be connected or accepted before data can be sent or received on it.
  explicit basic_stream_socket(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
  }

  /// Destructor.
  ~basic_stream_socket()
  {
    service_.destroy(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Close the socket.
  void close()
  {
    service_.destroy(impl_);
  }

  /// Get a reference to the lowest layer.
  lowest_layer_type& lowest_layer()
  {
    return *this;
  }

  /// Get the underlying implementation in the native type.
  impl_type impl()
  {
    return impl_;
  }

  /// Set the underlying implementation in the native type. The object must not
  /// be open priot to this call being made.
  void set_impl(impl_type new_impl)
  {
    service_.create(impl_, new_impl);
  }

  /// Send the given data to the peer. Returns the number of bytes sent or
  /// 0 if the connection was closed cleanly. Throws a socket_error exception
  /// on failure.
  size_t send(const void* data, size_t length)
  {
    return service_.send(impl_, data, length);
  }

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Handler>
  void async_send(const void* data, size_t length, Handler handler)
  {
    service_.async_send(impl_, data, length, handler,
        null_completion_context::instance());
  }

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Handler, typename Completion_Context>
  void async_send(const void* data, size_t length, Handler handler,
      Completion_Context& context)
  {
    service_.async_send(impl_, data, length, handler, context);
  }

  /// Receive some data from the peer. Returns the number of bytes received or
  /// 0 if the connection was closed cleanly. Throws a socket_error exception
  /// on failure.
  size_t recv(void* data, size_t max_length)
  {
    return service_.recv(impl_, data, max_length);
  }

  /// Start an asynchronous receive. The buffer for the data being received must
  /// be valid for the lifetime of the asynchronous operation.
  template <typename Handler>
  void async_recv(void* data, size_t max_length, Handler handler)
  {
    service_.async_recv(impl_, data, max_length, handler,
        null_completion_context::instance());
  }

  /// Start an asynchronous receive. The buffer for the data being received must
  /// be valid for the lifetime of the asynchronous operation.
  template <typename Handler, typename Completion_Context>
  void async_recv(void* data, size_t max_length, Handler handler,
      Completion_Context& context)
  {
    service_.async_recv(impl_, data, max_length, handler, context);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_STREAM_SOCKET_HPP
