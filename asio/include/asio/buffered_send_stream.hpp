//
// buffered_send_stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BUFFERED_SEND_STREAM_HPP
#define ASIO_BUFFERED_SEND_STREAM_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstring>
#include <boost/noncopyable.hpp>
#include <boost/type_traits.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_send_stream_fwd.hpp"
#include "asio/send.hpp"

namespace asio {

/// Adds buffering to the send-related operations of a stream.
/**
 * The buffered_send_stream class template can be used to add buffering to the
 * synchronous and asynchronous send operations of a stream.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object, Async_Recv_Stream, Async_Send_Stream, Stream,
 * Sync_Recv_Stream, Sync_Send_Stream.
 */
template <typename Stream, typename Buffer>
class buffered_send_stream
  : private boost::noncopyable
{
public:
  /// Construct, passing the specified argument to initialise the next layer.
  template <typename Arg>
  explicit buffered_send_stream(Arg& a)
    : next_layer_(a),
      buffer_()
  {
  }

  /// The type of the next layer.
  typedef typename boost::remove_reference<Stream>::type next_layer_type;

  /// Get a reference to the next layer.
  next_layer_type& next_layer()
  {
    return next_layer_;
  }

  /// The type of the lowest layer.
  typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

  /// Get a reference to the lowest layer.
  lowest_layer_type& lowest_layer()
  {
    return next_layer_.lowest_layer();
  }

  /// The demuxer type for this asynchronous type.
  typedef typename next_layer_type::demuxer_type demuxer_type;

  /// Get the demuxer associated with the asynchronous object.
  demuxer_type& demuxer()
  {
    return next_layer_.demuxer();
  }

  /// The buffer type for this buffering layer.
  typedef Buffer buffer_type;

  /// Get the send buffer used by this buffering layer.
  buffer_type& send_buffer()
  {
    return buffer_;
  }

  /// Close the stream.
  void close()
  {
    next_layer_.close();
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last send operation, or 0 if the
  /// underlying connection was closed. Throws an exception on failure.
  size_t flush()
  {
    size_t total_bytes_sent = 0;
    size_t last_bytes_sent = send_n(next_layer_, buffer_.begin(),
        buffer_.size(), &total_bytes_sent);
    buffer_.pop(total_bytes_sent);
    return last_bytes_sent;
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last send operation, or 0 if the
  /// underlying connection was closed.
  template <typename Error_Handler>
  size_t flush(Error_Handler error_handler)
  {
    size_t total_bytes_sent = 0;
    size_t last_bytes_sent = send_n(next_layer_, buffer_.begin(),
        buffer_.size(), &total_bytes_sent, error_handler);
    buffer_.pop(total_bytes_sent);
    return last_bytes_sent;
  }

  template <typename Handler>
  class flush_handler
  {
  public:
    flush_handler(buffered_send_stream<Stream, Buffer>& stream,
        Handler handler)
      : stream_(stream),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t last_bytes_sent,
        size_t total_bytes_sent)
    {
      stream_.send_buffer().pop(total_bytes_sent);
      stream_.demuxer().dispatch(
          detail::bind_handler(handler_, e, last_bytes_sent));
    }

  private:
    buffered_send_stream<Stream, Buffer>& stream_;
    Handler handler_;
  };

  /// Start an asynchronous flush.
  template <typename Handler>
  void async_flush(Handler handler)
  {
    async_send_n(next_layer_, buffer_.begin(), buffer_.size(),
        flush_handler<Handler>(*this, handler));
  }

  /// Send the given data to the peer. Returns the number of bytes sent or 0 if
  /// the stream was closed cleanly. Throws an exception on failure.
  size_t send(const void* data, size_t length)
  {
    if (buffer_.size() == buffer_.capacity() && !flush())
      return 0;
    return copy(data, length);
  }

  /// Send the given data to the peer. Returns the number of bytes sent or 0 if
  /// the stream was closed cleanly.
  template <typename Error_Handler>
  size_t send(const void* data, size_t length, Error_Handler error_handler)
  {
    if (buffer_.size() == buffer_.capacity() && !flush(error_handler))
      return 0;
    return copy(data, length);
  }

  template <typename Handler>
  class send_handler
  {
  public:
    send_handler(buffered_send_stream<Stream, Buffer>& stream,
        const void* data, size_t length, Handler handler)
      : stream_(stream),
        data_(data),
        length_(length),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_sent)
    {
      if (e || bytes_sent == 0)
      {
        size_t length = 0;
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e, length));
      }
      else
      {
        using namespace std; // For memcpy.
        size_t orig_size = stream_.send_buffer().size();
        size_t bytes_avail = stream_.send_buffer().capacity() - orig_size;
        size_t bytes_copied = (length_ < bytes_avail) ? length_ : bytes_avail;
        stream_.send_buffer().resize(orig_size + bytes_copied);
        memcpy(stream_.send_buffer().begin() + orig_size, data_, bytes_copied);
        stream_.demuxer().dispatch(
            detail::bind_handler(handler_, e, bytes_copied));
      }
    }

  private:
    buffered_send_stream<Stream, Buffer>& stream_;
    const void* data_;
    size_t length_;
    Handler handler_;
  };

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Handler>
  void async_send(const void* data, size_t length, Handler handler)
  {
    if (buffer_.size() == buffer_.capacity())
    {
      async_flush(send_handler<Handler>(*this, data, length, handler));
    }
    else
    {
      size_t bytes_copied = copy(data, length);
      next_layer_.demuxer().post(
          detail::bind_handler(handler, 0, bytes_copied));
    }
  }

  /// Receive some data from the peer. Returns the number of bytes received or
  /// 0 if the stream was closed cleanly. Throws an exception on failure.
  size_t recv(void* data, size_t max_length)
  {
    return next_layer_.recv(data, max_length);
  }

  /// Receive some data from the peer. Returns the number of bytes received or
  /// 0 if the stream was closed cleanly.
  template <typename Error_Handler>
  size_t recv(void* data, size_t max_length, Error_Handler error_handler)
  {
    return next_layer_.recv(data, max_length, error_handler);
  }

  /// Start an asynchronous receive. The buffer for the data being received
  /// must be valid for the lifetime of the asynchronous operation.
  template <typename Handler>
  void async_recv(void* data, size_t max_length, Handler handler)
  {
    next_layer_.async_recv(data, max_length, handler);
  }

private:
  /// Copy data into the internal buffer from the specified source buffer.
  /// Returns the number of bytes copied.
  size_t copy(const void* data, size_t length)
  {
    using namespace std; // For memcpy.

    size_t orig_size = buffer_.size();
    size_t bytes_avail = buffer_.capacity() - orig_size;
    size_t bytes_copied = (length < bytes_avail) ? length : bytes_avail;
    buffer_.resize(orig_size + bytes_copied);
    memcpy(buffer_.begin() + orig_size, data, bytes_copied);

    return length;
  }

  /// The next layer.
  Stream next_layer_;

  // The data in the buffer.
  Buffer buffer_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_SEND_STREAM_HPP
