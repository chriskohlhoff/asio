//
// buffered_recv_stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_BUFFERED_RECV_STREAM_HPP
#define ASIO_BUFFERED_RECV_STREAM_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstring>
#include <boost/noncopyable.hpp>
#include <boost/type_traits.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_recv_stream_fwd.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_resize_guard.hpp"

namespace asio {

/// The buffered_recv_stream class template can be used to add buffering to the
/// recv-related operations of a stream.
template <typename Next_Layer, typename Buffer>
class buffered_recv_stream
  : private boost::noncopyable
{
public:
  /// Construct, passing the specified demuxer to initialise the next layer.
  template <typename Arg>
  explicit buffered_recv_stream(Arg& a)
    : next_layer_(a),
      buffer_()
  {
  }

  /// The type of the next layer.
  typedef typename boost::remove_reference<Next_Layer>::type next_layer_type;

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

  /// Get the recv buffer used by this buffering layer.
  buffer_type& recv_buffer()
  {
    return buffer_;
  }

  /// Close the stream.
  void close()
  {
    next_layer_.close();
  }

  /// Send the given data to the peer. Returns the number of bytes sent or 0 if
  /// the stream was closed cleanly. Throws an exception on failure.
  size_t send(const void* data, size_t length)
  {
    return next_layer_.send(data, length);
  }

  /// Send the given data to the peer. Returns the number of bytes sent or 0 if
  /// the stream was closed cleanly.
  template <typename Error_Handler>
  size_t send(const void* data, size_t length, Error_Handler error_handler)
  {
    return next_layer_.send(data, length, error_handler);
  }

  /// Start an asynchronous send. The data being sent must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Handler>
  void async_send(const void* data, size_t length, Handler handler)
  {
    next_layer_.async_send(data, length, handler);
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if the underlying connection
  /// was closed. Throws an exception on failure.
  size_t fill()
  {
    detail::buffer_resize_guard<Buffer> resize_guard(buffer_);
    size_t previous_size = buffer_.size();
    buffer_.resize(buffer_.capacity());
    buffer_.resize(previous_size + next_layer_.recv(
          buffer_.begin() + previous_size, buffer_.size() - previous_size));
    resize_guard.commit();
    return buffer_.size() - previous_size;
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if the underlying connection
  /// was closed.
  template <typename Error_Handler>
  size_t fill(Error_Handler error_handler)
  {
    detail::buffer_resize_guard<Buffer> resize_guard(buffer_);
    size_t previous_size = buffer_.size();
    buffer_.resize(buffer_.capacity());
    buffer_.resize(previous_size + next_layer_.recv(
          buffer_.begin() + previous_size, buffer_.size() - previous_size,
          error_handler));
    resize_guard.commit();
    return buffer_.size() - previous_size;
  }

  template <typename Handler>
  class fill_handler
  {
  public:
    fill_handler(buffered_recv_stream<Next_Layer, Buffer>& stream,
        size_t previous_size, Handler handler)
      : stream_(stream),
        previous_size_(previous_size),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_recvd)
    {
      stream_.recv_buffer().resize(previous_size_ + bytes_recvd);
      stream_.demuxer().dispatch(
          detail::bind_handler(handler_, e, bytes_recvd));
    }

  private:
    buffered_recv_stream<Next_Layer, Buffer>& stream_;
    size_t previous_size_;
    Handler handler_;
  };

  /// Start an asynchronous fill.
  template <typename Handler>
  void async_fill(Handler handler)
  {
    size_t previous_size = buffer_.size();
    buffer_.resize(buffer_.capacity());
    next_layer_.async_recv(buffer_.begin() + previous_size,
        buffer_.size() - previous_size,
        fill_handler<Handler>(*this, previous_size, handler));
  }

  /// Receive some data from the peer. Returns the number of bytes received or
  /// 0 if the stream was closed cleanly. Throws an exception on failure.
  size_t recv(void* data, size_t max_length)
  {
    if (buffer_.empty() && !fill())
      return 0;
    return copy(data, max_length);
  }

  /// Receive some data from the peer. Returns the number of bytes received or
  /// 0 if the stream was closed cleanly.
  template <typename Error_Handler>
  size_t recv(void* data, size_t max_length, Error_Handler error_handler)
  {
    if (buffer_.empty() && !fill(error_handler))
      return 0;
    return copy(data, max_length);
  }

  template <typename Handler>
  class recv_handler
  {
  public:
    recv_handler(buffered_recv_stream<Next_Layer, Buffer>& stream, void* data,
        size_t max_length, Handler handler)
      : stream_(stream),
        data_(data),
        max_length_(max_length),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_recvd)
    {
      if (e || stream_.recv_buffer().empty())
      {
        size_t length = 0;
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e, length));
      }
      else
      {
        using namespace std; // For memcpy.
        size_t bytes_avail = stream_.recv_buffer().size();
        size_t length = (max_length_ < bytes_avail)
          ? max_length_ : bytes_avail;
        memcpy(data_, stream_.recv_buffer().begin(), length);
        stream_.recv_buffer().pop(length);
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e, length));
      }
    }

  private:
    buffered_recv_stream<Next_Layer, Buffer>& stream_;
    void* data_;
    size_t max_length_;
    Handler handler_;
  };

  /// Start an asynchronous receive. The buffer for the data being received
  /// must be valid for the lifetime of the asynchronous operation.
  template <typename Handler>
  void async_recv(void* data, size_t max_length, Handler handler)
  {
    if (buffer_.empty())
    {
      async_fill(recv_handler<Handler>(*this, data, max_length, handler));
    }
    else
    {
      size_t length = copy(data, max_length);
      next_layer_.demuxer().post(detail::bind_handler(handler, 0, length));
    }
  }

private:
  /// Copy data out of the internal buffer to the specified target buffer.
  /// Returns the number of bytes copied.
  size_t copy(void* data, size_t max_length)
  {
    using namespace std; // For memcpy.

    size_t bytes_avail = buffer_.size();
    size_t length = (max_length < bytes_avail) ? max_length : bytes_avail;
    memcpy(data, buffer_.begin(), length);
    buffer_.pop(length);

    return length;
  }

  /// The next layer.
  Next_Layer next_layer_;

  // The data in the buffer.
  Buffer buffer_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_RECV_STREAM_HPP
