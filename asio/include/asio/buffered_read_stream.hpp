//
// buffered_read_stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BUFFERED_READ_STREAM_HPP
#define ASIO_BUFFERED_READ_STREAM_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstring>
#include <boost/noncopyable.hpp>
#include <boost/type_traits.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_read_stream_fwd.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_resize_guard.hpp"

namespace asio {

/// Adds buffering to the read-related operations of a stream.
/**
 * The buffered_read_stream class template can be used to add buffering to the
 * synchronous and asynchronous read operations of a stream.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object, Async_Read_Stream, Async_Write_Stream, Stream,
 * Sync_Read_Stream, Sync_Write_Stream.
 */
template <typename Stream, typename Buffer>
class buffered_read_stream
  : private boost::noncopyable
{
public:
  /// Construct, passing the specified demuxer to initialise the next layer.
  template <typename Arg>
  explicit buffered_read_stream(Arg& a)
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

  /// Get the read buffer used by this buffering layer.
  buffer_type& read_buffer()
  {
    return buffer_;
  }

  /// Close the stream.
  void close()
  {
    next_layer_.close();
  }

  /// Write the given data to the stream. Returns the number of bytes written or
  /// 0 if the stream was closed cleanly. Throws an exception on failure.
  size_t write(const void* data, size_t length)
  {
    return next_layer_.write(data, length);
  }

  /// Write the given data to the stream. Returns the number of bytes written or
  /// 0 if the stream was closed cleanly.
  template <typename Error_Handler>
  size_t write(const void* data, size_t length, Error_Handler error_handler)
  {
    return next_layer_.write(data, length, error_handler);
  }

  /// Start an asynchronous write. The data being written must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Handler>
  void async_write(const void* data, size_t length, Handler handler)
  {
    next_layer_.async_write(data, length, handler);
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if the underlying stream was
  /// closed. Throws an exception on failure.
  size_t fill()
  {
    detail::buffer_resize_guard<Buffer> resize_guard(buffer_);
    size_t previous_size = buffer_.size();
    buffer_.resize(buffer_.capacity());
    buffer_.resize(previous_size + next_layer_.read(
          buffer_.begin() + previous_size, buffer_.size() - previous_size));
    resize_guard.commit();
    return buffer_.size() - previous_size;
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if the underlying stream was
  /// closed.
  template <typename Error_Handler>
  size_t fill(Error_Handler error_handler)
  {
    detail::buffer_resize_guard<Buffer> resize_guard(buffer_);
    size_t previous_size = buffer_.size();
    buffer_.resize(buffer_.capacity());
    buffer_.resize(previous_size + next_layer_.read(
          buffer_.begin() + previous_size, buffer_.size() - previous_size,
          error_handler));
    resize_guard.commit();
    return buffer_.size() - previous_size;
  }

  template <typename Handler>
  class fill_handler
  {
  public:
    fill_handler(buffered_read_stream<Stream, Buffer>& stream,
        size_t previous_size, Handler handler)
      : stream_(stream),
        previous_size_(previous_size),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_readd)
    {
      stream_.read_buffer().resize(previous_size_ + bytes_readd);
      stream_.demuxer().dispatch(
          detail::bind_handler(handler_, e, bytes_readd));
    }

  private:
    buffered_read_stream<Stream, Buffer>& stream_;
    size_t previous_size_;
    Handler handler_;
  };

  /// Start an asynchronous fill.
  template <typename Handler>
  void async_fill(Handler handler)
  {
    size_t previous_size = buffer_.size();
    buffer_.resize(buffer_.capacity());
    next_layer_.async_read(buffer_.begin() + previous_size,
        buffer_.size() - previous_size,
        fill_handler<Handler>(*this, previous_size, handler));
  }

  /// Read some data from the stream. Returns the number of bytes read or 0 if
  /// the stream was closed cleanly. Throws an exception on failure.
  size_t read(void* data, size_t max_length)
  {
    if (buffer_.empty() && !fill())
      return 0;
    return copy(data, max_length);
  }

  /// Read some data from the stream. Returns the number of bytes read or 0 if
  /// the stream was closed cleanly.
  template <typename Error_Handler>
  size_t read(void* data, size_t max_length, Error_Handler error_handler)
  {
    if (buffer_.empty() && !fill(error_handler))
      return 0;
    return copy(data, max_length);
  }

  template <typename Handler>
  class read_handler
  {
  public:
    read_handler(buffered_read_stream<Stream, Buffer>& stream, void* data,
        size_t max_length, Handler handler)
      : stream_(stream),
        data_(data),
        max_length_(max_length),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_readd)
    {
      if (e || stream_.read_buffer().empty())
      {
        size_t length = 0;
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e, length));
      }
      else
      {
        using namespace std; // For memcpy.
        size_t bytes_avail = stream_.read_buffer().size();
        size_t length = (max_length_ < bytes_avail)
          ? max_length_ : bytes_avail;
        memcpy(data_, stream_.read_buffer().begin(), length);
        stream_.read_buffer().pop(length);
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e, length));
      }
    }

  private:
    buffered_read_stream<Stream, Buffer>& stream_;
    void* data_;
    size_t max_length_;
    Handler handler_;
  };

  /// Start an asynchronous read. The buffer into which the data will be read
  /// must be valid for the lifetime of the asynchronous operation.
  template <typename Handler>
  void async_read(void* data, size_t max_length, Handler handler)
  {
    if (buffer_.empty())
    {
      async_fill(read_handler<Handler>(*this, data, max_length, handler));
    }
    else
    {
      size_t length = copy(data, max_length);
      next_layer_.demuxer().post(detail::bind_handler(handler, 0, length));
    }
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read
  /// or 0 if the stream was closed cleanly.
  size_t peek(void* data, size_t max_length)
  {
    if (buffer_.empty() && !fill())
      return 0;
    return peek_copy(data, max_length);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read
  /// or 0 if the stream was closed cleanly.
  template <typename Error_Handler>
  size_t peek(void* data, size_t max_length, Error_Handler error_handler)
  {
    if (buffer_.empty() && !fill(error_handler))
      return 0;
    return peek_copy(data, max_length);
  }

  /// Determine the amount of data that may be read without blocking.
  size_t in_avail()
  {
    return buffer_.size();
  }

  /// Determine the amount of data that may be read without blocking.
  template <typename Error_Handler>
  size_t in_avail(Error_Handler error_handler)
  {
    return buffer_.size();
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

  /// Copy data from the internal buffer to the specified target buffer, without
  /// removing the data from the internal buffer. Returns the number of bytes
  /// copied.
  size_t peek_copy(void* data, size_t max_length)
  {
    using namespace std; // For memcpy.

    size_t bytes_avail = buffer_.size();
    size_t length = (max_length < bytes_avail) ? max_length : bytes_avail;
    memcpy(data, buffer_.begin(), length);

    return length;
  }

  /// The next layer.
  Stream next_layer_;

  // The data in the buffer.
  Buffer buffer_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_READ_STREAM_HPP
