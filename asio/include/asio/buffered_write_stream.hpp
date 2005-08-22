//
// buffered_write_stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BUFFERED_WRITE_STREAM_HPP
#define ASIO_BUFFERED_WRITE_STREAM_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstring>
#include <boost/noncopyable.hpp>
#include <boost/type_traits.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_write_stream_fwd.hpp"
#include "asio/write.hpp"

namespace asio {

/// Adds buffering to the write-related operations of a stream.
/**
 * The buffered_write_stream class template can be used to add buffering to the
 * synchronous and asynchronous write operations of a stream.
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
class buffered_write_stream
  : private boost::noncopyable
{
public:
  /// Construct, passing the specified argument to initialise the next layer.
  template <typename Arg>
  explicit buffered_write_stream(Arg& a)
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

  /// Get the write buffer used by this buffering layer.
  buffer_type& write_buffer()
  {
    return buffer_;
  }

  /// Close the stream.
  void close()
  {
    next_layer_.close();
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last write operation, or 0 if the
  /// underlying stream was closed. Throws an exception on failure.
  size_t flush()
  {
    size_t total_bytes_written = 0;
    size_t last_bytes_written = write_n(next_layer_, buffer_.begin(),
        buffer_.size(), &total_bytes_written);
    buffer_.pop(total_bytes_written);
    return last_bytes_written;
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last write operation, or 0 if the
  /// underlying stream was closed.
  template <typename Error_Handler>
  size_t flush(Error_Handler error_handler)
  {
    size_t total_bytes_written = 0;
    size_t last_bytes_written = write_n(next_layer_, buffer_.begin(),
        buffer_.size(), &total_bytes_written, error_handler);
    buffer_.pop(total_bytes_written);
    return last_bytes_written;
  }

  template <typename Handler>
  class flush_handler
  {
  public:
    flush_handler(buffered_write_stream<Stream, Buffer>& stream,
        Handler handler)
      : stream_(stream),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t last_bytes_written,
        size_t total_bytes_written)
    {
      stream_.write_buffer().pop(total_bytes_written);
      stream_.demuxer().dispatch(
          detail::bind_handler(handler_, e, last_bytes_written));
    }

  private:
    buffered_write_stream<Stream, Buffer>& stream_;
    Handler handler_;
  };

  /// Start an asynchronous flush.
  template <typename Handler>
  void async_flush(Handler handler)
  {
    async_write_n(next_layer_, buffer_.begin(), buffer_.size(),
        flush_handler<Handler>(*this, handler));
  }

  /// Write the given data to the stream. Returns the number of bytes written or
  /// 0 if the stream was closed cleanly. Throws an exception on failure.
  size_t write(const void* data, size_t length)
  {
    if (buffer_.size() == buffer_.capacity() && !flush())
      return 0;
    return copy(data, length);
  }

  /// Write the given data to the stream. Returns the number of bytes written or
  /// 0 if the stream was closed cleanly.
  template <typename Error_Handler>
  size_t write(const void* data, size_t length, Error_Handler error_handler)
  {
    if (buffer_.size() == buffer_.capacity() && !flush(error_handler))
      return 0;
    return copy(data, length);
  }

  template <typename Handler>
  class write_handler
  {
  public:
    write_handler(buffered_write_stream<Stream, Buffer>& stream,
        const void* data, size_t length, Handler handler)
      : stream_(stream),
        data_(data),
        length_(length),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_written)
    {
      if (e || bytes_written == 0)
      {
        size_t length = 0;
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e, length));
      }
      else
      {
        using namespace std; // For memcpy.
        size_t orig_size = stream_.write_buffer().size();
        size_t bytes_avail = stream_.write_buffer().capacity() - orig_size;
        size_t bytes_copied = (length_ < bytes_avail) ? length_ : bytes_avail;
        stream_.write_buffer().resize(orig_size + bytes_copied);
        memcpy(stream_.write_buffer().begin() + orig_size, data_, bytes_copied);
        stream_.demuxer().dispatch(
            detail::bind_handler(handler_, e, bytes_copied));
      }
    }

  private:
    buffered_write_stream<Stream, Buffer>& stream_;
    const void* data_;
    size_t length_;
    Handler handler_;
  };

  /// Start an asynchronous write. The data being written must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Handler>
  void async_write(const void* data, size_t length, Handler handler)
  {
    if (buffer_.size() == buffer_.capacity())
    {
      async_flush(write_handler<Handler>(*this, data, length, handler));
    }
    else
    {
      size_t bytes_copied = copy(data, length);
      next_layer_.demuxer().post(
          detail::bind_handler(handler, 0, bytes_copied));
    }
  }

  /// Read some data from the stream. Returns the number of bytes read or 0 if
  /// the stream was closed cleanly. Throws an exception on failure.
  size_t read(void* data, size_t max_length)
  {
    return next_layer_.read(data, max_length);
  }

  /// Read some data from the stream. Returns the number of bytes read or 0 if
  /// the stream was closed cleanly.
  template <typename Error_Handler>
  size_t read(void* data, size_t max_length, Error_Handler error_handler)
  {
    return next_layer_.read(data, max_length, error_handler);
  }

  /// Start an asynchronous read. The buffer into which the data will be read
  /// must be valid for the lifetime of the asynchronous operation.
  template <typename Handler>
  void async_read(void* data, size_t max_length, Handler handler)
  {
    next_layer_.async_read(data, max_length, handler);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read
  /// or 0 if the stream was closed cleanly.
  size_t peek(void* data, size_t max_length)
  {
    return next_layer_.peek(data, max_length);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read
  /// or 0 if the stream was closed cleanly.
  template <typename Error_Handler>
  size_t peek(void* data, size_t max_length, Error_Handler error_handler)
  {
    return next_layer_.peek(data, max_length, error_handler);
  }

  /// Determine the amount of data that may be read without blocking.
  size_t in_avail()
  {
    return next_layer_.in_avail();
  }

  /// Determine the amount of data that may be read without blocking.
  template <typename Error_Handler>
  size_t in_avail(Error_Handler error_handler)
  {
    return next_layer_.in_avail(error_handler);
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

#endif // ASIO_BUFFERED_WRITE_STREAM_HPP
