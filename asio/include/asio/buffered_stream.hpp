//
// buffered_stream.hpp
// ~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_BUFFERED_STREAM_HPP
#define ASIO_BUFFERED_STREAM_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_recv_stream.hpp"
#include "asio/buffered_send_stream.hpp"
#include "asio/buffered_stream_fwd.hpp"

namespace asio {

/// The buffered_stream class template can be used to add buffering to both the
/// send- and recv- related operations of a stream.
template <typename Next_Layer, typename Buffer>
class buffered_stream
  : private boost::noncopyable
{
public:
  /// Construct, passing the specified argument to initialise the next layer.
  template <typename Arg>
  explicit buffered_stream(Arg& a)
    : stream_impl_(a)
  {
  }

  /// The type of the next layer.
  typedef typename boost::remove_reference<Next_Layer>::type next_layer_type;

  /// Get a reference to the next layer.
  next_layer_type& next_layer()
  {
    return stream_impl_.next_layer().next_layer();
  }

  /// The type of the lowest layer.
  typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

  /// Get a reference to the lowest layer.
  lowest_layer_type& lowest_layer()
  {
    return stream_impl_.lowest_layer();
  }

  /// The demuxer type for this asynchronous type.
  typedef typename next_layer_type::demuxer_type demuxer_type;

  /// Get the demuxer associated with the asynchronous object.
  demuxer_type& demuxer()
  {
    return stream_impl_.demuxer();
  }

  /// The buffer type for this buffering layer.
  typedef Buffer buffer_type;

  /// Get the recv buffer used by this buffering layer.
  buffer_type& recv_buffer()
  {
    return stream_impl_.recv_buffer();
  }

  /// Get the send buffer used by this buffering layer.
  buffer_type& send_buffer()
  {
    return stream_impl_.next_layer().send_buffer();
  }

  /// Close the stream.
  void close()
  {
    stream_impl_.close();
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last send operation, or 0 if the
  /// underlying connection was closed. Throws an exception on failure.
  size_t flush()
  {
    return stream_impl_.next_layer().flush();
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last send operation, or 0 if the
  /// underlying connection was closed.
  template <typename Error_Handler>
  size_t flush(Error_Handler error_handler)
  {
    return stream_impl_.next_layer().flush(error_handler);
  }

  /// Start an asynchronous flush.
  template <typename Handler>
  void async_flush(Handler handler)
  {
    return stream_impl_.next_layer().async_flush(handler);
  }

  /// Send the given data to the peer. Returns the number of bytes sent or 0 if
  /// the stream was closed cleanly. Throws an exception on failure.
  size_t send(const void* data, size_t length)
  {
    return stream_impl_.send(data, length);
  }

  /// Send the given data to the peer. Returns the number of bytes sent or 0 if
  /// the stream was closed cleanly.
  template <typename Error_Handler>
  size_t send(const void* data, size_t length, Error_Handler error_handler)
  {
    return stream_impl_.send(data, length, error_handler);
  }

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Handler>
  void async_send(const void* data, size_t length, Handler handler)
  {
    stream_impl_.async_send(data, length, handler);
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if the underlying connection
  /// was closed. Throws an exception on failure.
  size_t fill()
  {
    return stream_impl_.fill();
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if the underlying connection
  /// was closed.
  template <typename Error_Handler>
  size_t fill(Error_Handler error_handler)
  {
    return stream_impl_.fill(error_handler);
  }

  /// Start an asynchronous fill.
  template <typename Handler>
  void async_fill(Handler handler)
  {
    stream_impl_.async_fill(handler);
  }

  /// Receive some data from the peer. Returns the number of bytes received or
  /// 0 if the stream was closed cleanly. Throws an exception on failure.
  size_t recv(void* data, size_t max_length)
  {
    return stream_impl_.recv(data, max_length);
  }

  /// Receive some data from the peer. Returns the number of bytes received or
  /// 0 if the stream was closed cleanly.
  template <typename Error_Handler>
  size_t recv(void* data, size_t max_length, Error_Handler error_handler)
  {
    return stream_impl_.recv(data, max_length, error_handler);
  }

  /// Start an asynchronous receive. The buffer for the data being received
  /// must be valid for the lifetime of the asynchronous operation.
  template <typename Handler>
  void async_recv(void* data, size_t max_length, Handler handler)
  {
    stream_impl_.async_recv(data, max_length, handler);
  }

private:
  /// The buffered stream implementation.
  buffered_recv_stream<buffered_send_stream<Next_Layer, Buffer>, Buffer>
    stream_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_STREAM_HPP
