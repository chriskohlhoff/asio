//
// buffered_stream.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BUFFERED_STREAM_HPP
#define ASIO_BUFFERED_STREAM_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_read_stream.hpp"
#include "asio/buffered_write_stream.hpp"
#include "asio/buffered_stream_fwd.hpp"

namespace asio {

/// Adds buffering to the read- and write-related operations of a stream.
/**
 * The buffered_stream class template can be used to add buffering to the
 * synchronous and asynchronous read and write operations of a stream.
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
  typedef typename boost::remove_reference<Stream>::type next_layer_type;

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

  /// Get the read buffer used by this buffering layer.
  buffer_type& read_buffer()
  {
    return stream_impl_.read_buffer();
  }

  /// Get the write buffer used by this buffering layer.
  buffer_type& write_buffer()
  {
    return stream_impl_.next_layer().write_buffer();
  }

  /// Close the stream.
  void close()
  {
    stream_impl_.close();
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last write operation, or 0 if the
  /// underlying stream was closed. Throws an exception on failure.
  std::size_t flush()
  {
    return stream_impl_.next_layer().flush();
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last write operation, or 0 if the
  /// underlying stream was closed.
  template <typename Error_Handler>
  std::size_t flush(Error_Handler error_handler)
  {
    return stream_impl_.next_layer().flush(error_handler);
  }

  /// Start an asynchronous flush.
  template <typename Handler>
  void async_flush(Handler handler)
  {
    return stream_impl_.next_layer().async_flush(handler);
  }

  /// Write the given data to the stream. Returns the number of bytes written or
  /// 0 if the stream was closed cleanly. Throws an exception on failure.
  template <typename Const_Buffers>
  std::size_t write(const Const_Buffers& buffers)
  {
    return stream_impl_.write(buffers);
  }

  /// Write the given data to the stream. Returns the number of bytes written or
  /// 0 if the stream was closed cleanly.
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t write(const Const_Buffers& buffers, Error_Handler error_handler)
  {
    return stream_impl_.write(buffers, error_handler);
  }

  /// Start an asynchronous write. The data being written must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Handler>
  void async_write(const Const_Buffers& buffers, Handler handler)
  {
    stream_impl_.async_write(buffers, handler);
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if the underlying stream was
  /// closed. Throws an exception on failure.
  std::size_t fill()
  {
    return stream_impl_.fill();
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if the underlying stream was
  /// closed.
  template <typename Error_Handler>
  std::size_t fill(Error_Handler error_handler)
  {
    return stream_impl_.fill(error_handler);
  }

  /// Start an asynchronous fill.
  template <typename Handler>
  void async_fill(Handler handler)
  {
    stream_impl_.async_fill(handler);
  }

  /// Read some data from the stream. Returns the number of bytes read or 0 if
  /// the stream was closed cleanly. Throws an exception on failure.
  template <typename Mutable_Buffers>
  std::size_t read(const Mutable_Buffers& buffers)
  {
    return stream_impl_.read(buffers);
  }

  /// Read some data from the stream. Returns the number of bytes read or 0 if
  /// the stream was closed cleanly.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t read(const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    return stream_impl_.read(buffers, error_handler);
  }

  /// Start an asynchronous read. The buffer into which the data will be read
  /// must be valid for the lifetime of the asynchronous operation.
  template <typename Mutable_Buffers, typename Handler>
  void async_read(const Mutable_Buffers& buffers, Handler handler)
  {
    stream_impl_.async_read(buffers, handler);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read
  /// or 0 if the stream was closed cleanly.
  template <typename Mutable_Buffers>
  std::size_t peek(const Mutable_Buffers& buffers)
  {
    return stream_impl_.peek(buffers);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read
  /// or 0 if the stream was closed cleanly.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t peek(const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    return stream_impl_.peek(data, max_length, error_handler);
  }

  /// Determine the amount of data that may be read without blocking.
  std::size_t in_avail()
  {
    return stream_impl_.in_avail();
  }

  /// Determine the amount of data that may be read without blocking.
  template <typename Error_Handler>
  std::size_t in_avail(Error_Handler error_handler)
  {
    return stream_impl_.in_avail(error_handler);
  }

private:
  /// The buffered stream implementation.
  buffered_read_stream<buffered_write_stream<Stream, Buffer>, Buffer>
    stream_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_STREAM_HPP
