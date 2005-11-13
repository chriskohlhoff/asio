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
#include <cstddef>
#include <cstring>
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/type_traits.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_write_stream_fwd.hpp"
#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/write.hpp"
#include "asio/detail/bind_handler.hpp"

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
 * Async_Object, Async_Read_Stream, Async_Write_Stream, Error_Source, Stream,
 * Sync_Read_Stream, Sync_Write_Stream.
 */
template <typename Stream, typename Buffer>
class buffered_write_stream
  : private boost::noncopyable
{
public:
  /// The type of the next layer.
  typedef typename boost::remove_reference<Stream>::type next_layer_type;

  /// The type of the lowest layer.
  typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

  /// The demuxer type for this asynchronous type.
  typedef typename next_layer_type::demuxer_type demuxer_type;

  /// The type used for reporting errors.
  typedef typename next_layer_type::error_type error_type;

  /// The buffer type for this buffering layer.
  typedef Buffer buffer_type;

  /// Construct, passing the specified argument to initialise the next layer.
  template <typename Arg>
  explicit buffered_write_stream(Arg& a)
    : next_layer_(a),
      buffer_()
  {
  }

  /// Get a reference to the next layer.
  next_layer_type& next_layer()
  {
    return next_layer_;
  }

  /// Get a reference to the lowest layer.
  lowest_layer_type& lowest_layer()
  {
    return next_layer_.lowest_layer();
  }

  /// Get the demuxer associated with the asynchronous object.
  demuxer_type& demuxer()
  {
    return next_layer_.demuxer();
  }

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

  /// Close the stream.
  template <typename Error_Handler>
  void close(Error_Handler error_handler)
  {
    next_layer_.close(error_handler);
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last write operation. Throws an
  /// exception on failure.
  std::size_t flush()
  {
    std::size_t bytes_written = write(next_layer_,
        buffer(buffer_.begin(), buffer_.size()));
    buffer_.pop(bytes_written);
    return bytes_written;
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last write operation, or 0 if an
  /// error occurred and the error handler did not throw.
  template <typename Error_Handler>
  std::size_t flush(Error_Handler error_handler)
  {
    std::size_t bytes_written = write(next_layer_,
        buffer(buffer_.begin(), buffer_.size()), error_handler);
    buffer_.pop(bytes_written);
    return bytes_written;
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

    void operator()(const error_type& e, std::size_t bytes_written)
    {
      stream_.write_buffer().pop(bytes_written);
      stream_.demuxer().dispatch(
          detail::bind_handler(handler_, e, bytes_written));
    }

  private:
    buffered_write_stream<Stream, Buffer>& stream_;
    Handler handler_;
  };

  /// Start an asynchronous flush.
  template <typename Handler>
  void async_flush(Handler handler)
  {
    async_write(next_layer_, buffer(buffer_.begin(), buffer_.size()),
        flush_handler<Handler>(*this, handler));
  }

  /// Write the given data to the stream. Returns the number of bytes written.
  /// Throws an exception on failure.
  template <typename Const_Buffers>
  std::size_t write_some(const Const_Buffers& buffers)
  {
    if (buffer_.size() == buffer_.capacity())
      flush();
    return copy(buffers);
  }

  /// Write the given data to the stream. Returns the number of bytes written,
  /// or 0 if an error occurred and the error handler did not throw.
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t write_some(const Const_Buffers& buffers,
      Error_Handler error_handler)
  {
    if (buffer_.size() == buffer_.capacity() && !flush(error_handler))
      return 0;
    return copy(buffers);
  }

  template <typename Const_Buffers, typename Handler>
  class write_some_handler
  {
  public:
    write_some_handler(buffered_write_stream<Stream, Buffer>& stream,
        const Const_Buffers& buffers, Handler handler)
      : stream_(stream),
        buffers_(buffers),
        handler_(handler)
    {
    }

    void operator()(const error_type& e, std::size_t bytes_written)
    {
      if (e)
      {
        std::size_t length = 0;
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e, length));
      }
      else
      {
        using namespace std; // For memcpy.

        std::size_t orig_size = stream_.write_buffer().size();
        std::size_t space_avail = stream_.write_buffer().capacity() - orig_size;
        std::size_t bytes_copied = 0;

        typename Const_Buffers::const_iterator iter = buffers_.begin();
        typename Const_Buffers::const_iterator end = buffers_.end();
        for (; iter != end && space_avail > 0; ++iter)
        {
          std::size_t length = (iter->size() < space_avail)
            ? iter->size() : space_avail;
          stream_.write_buffer().resize(orig_size + bytes_copied + length);
          memcpy(stream_.write_buffer().begin() + orig_size + bytes_copied,
              iter->data(), length);
          bytes_copied += length;
          space_avail -= length;
        }

        stream_.demuxer().dispatch(
            detail::bind_handler(handler_, e, bytes_copied));
      }
    }

  private:
    buffered_write_stream<Stream, Buffer>& stream_;
    Const_Buffers buffers_;
    Handler handler_;
  };

  /// Start an asynchronous write. The data being written must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Handler>
  void async_write_some(const Const_Buffers& buffers, Handler handler)
  {
    if (buffer_.size() == buffer_.capacity())
    {
      async_flush(write_some_handler<Const_Buffers, Handler>(
            *this, buffers, handler));
    }
    else
    {
      std::size_t bytes_copied = copy(buffers);
      next_layer_.demuxer().post(
          detail::bind_handler(handler, 0, bytes_copied));
    }
  }

  /// Read some data from the stream. Returns the number of bytes read. Throws
  /// an exception on failure.
  template <typename Mutable_Buffers>
  std::size_t read_some(const Mutable_Buffers& buffers)
  {
    return next_layer_.read_some(buffers);
  }

  /// Read some data from the stream. Returns the number of bytes read or 0 if
  /// an error occurred and the error handler did not throw an exception.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t read_some(const Mutable_Buffers& buffers,
      Error_Handler error_handler)
  {
    return next_layer_.read_some(buffers, error_handler);
  }

  /// Start an asynchronous read. The buffer into which the data will be read
  /// must be valid for the lifetime of the asynchronous operation.
  template <typename Mutable_Buffers, typename Handler>
  void async_read_some(const Mutable_Buffers& buffers, Handler handler)
  {
    next_layer_.async_read_some(buffers, handler);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read.
  /// Throws an exception on failure.
  template <typename Mutable_Buffers>
  std::size_t peek(const Mutable_Buffers& buffers)
  {
    return next_layer_.peek(buffers);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read,
  /// or 0 if an error occurred and the error handler did not throw.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t peek(const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    return next_layer_.peek(buffers, error_handler);
  }

  /// Determine the amount of data that may be read without blocking.
  std::size_t in_avail()
  {
    return next_layer_.in_avail();
  }

  /// Determine the amount of data that may be read without blocking.
  template <typename Error_Handler>
  std::size_t in_avail(Error_Handler error_handler)
  {
    return next_layer_.in_avail(error_handler);
  }

private:
  /// Copy data into the internal buffer from the specified source buffer.
  /// Returns the number of bytes copied.
  template <typename Const_Buffers>
  std::size_t copy(const Const_Buffers& buffers)
  {
    using namespace std; // For memcpy.

    std::size_t orig_size = buffer_.size();
    std::size_t space_avail = buffer_.capacity() - orig_size;
    std::size_t bytes_copied = 0;

    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    for (; iter != end && space_avail > 0; ++iter)
    {
      std::size_t length = (iter->size() < space_avail)
        ? iter->size() : space_avail;
      buffer_.resize(orig_size + bytes_copied + length);
      memcpy(buffer_.begin() + orig_size + bytes_copied, iter->data(), length);
      bytes_copied += length;
      space_avail -= length;
    }

    return bytes_copied;
  }

  /// The next layer.
  Stream next_layer_;

  // The data in the buffer.
  Buffer buffer_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_WRITE_STREAM_HPP
