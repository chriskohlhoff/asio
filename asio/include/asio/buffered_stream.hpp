//
// buffered_stream.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_read_stream.hpp"
#include "asio/buffered_write_stream.hpp"
#include "asio/buffered_stream_fwd.hpp"
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/noncopyable.hpp"

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
 * Async_Object, Async_Read_Stream, Async_Write_Stream, Error_Source, Stream,
 * Sync_Read_Stream, Sync_Write_Stream.
 */
template <typename Stream>
class buffered_stream
  : private noncopyable
{
public:
  /// The type of the next layer.
  typedef typename boost::remove_reference<Stream>::type next_layer_type;

  /// The type of the lowest layer.
  typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

  /// The type used for reporting errors.
  typedef typename next_layer_type::error_type error_type;

  /// Construct, passing the specified argument to initialise the next layer.
  template <typename Arg>
  explicit buffered_stream(Arg& a)
    : inner_stream_impl_(a),
      stream_impl_(inner_stream_impl_)
  {
  }

  /// Construct, passing the specified argument to initialise the next layer.
  template <typename Arg>
  explicit buffered_stream(Arg& a, std::size_t read_buffer_size,
      std::size_t write_buffer_size)
    : inner_stream_impl_(a, write_buffer_size),
      stream_impl_(inner_stream_impl_, read_buffer_size)
  {
  }

  /// Get a reference to the next layer.
  next_layer_type& next_layer()
  {
    return stream_impl_.next_layer().next_layer();
  }

  /// Get a reference to the lowest layer.
  lowest_layer_type& lowest_layer()
  {
    return stream_impl_.lowest_layer();
  }

  /// Get the io_service associated with the object.
  asio::io_service& io_service()
  {
    return stream_impl_.io_service();
  }

  /// Close the stream.
  void close()
  {
    stream_impl_.close();
  }

  /// Close the stream.
  template <typename Error_Handler>
  void close(Error_Handler error_handler)
  {
    stream_impl_.close(error_handler);
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last write operation. Throws an
  /// exception on failure.
  std::size_t flush()
  {
    return stream_impl_.next_layer().flush();
  }

  /// Flush all data from the buffer to the next layer. Returns the number of
  /// bytes written to the next layer on the last write operation, or 0 if an
  /// error occurred and the error handler did not throw.
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

  /// Write the given data to the stream. Returns the number of bytes written.
  /// Throws an exception on failure.
  template <typename Const_Buffers>
  std::size_t write_some(const Const_Buffers& buffers)
  {
    return stream_impl_.write_some(buffers);
  }

  /// Write the given data to the stream. Returns the number of bytes written,
  /// or 0 if an error occurred and the error handler did not throw.
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t write_some(const Const_Buffers& buffers,
      Error_Handler error_handler)
  {
    return stream_impl_.write_some(buffers, error_handler);
  }

  /// Start an asynchronous write. The data being written must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Handler>
  void async_write_some(const Const_Buffers& buffers, Handler handler)
  {
    stream_impl_.async_write_some(buffers, handler);
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation. Throws an exception on failure.
  std::size_t fill()
  {
    return stream_impl_.fill();
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if an error occurred and the
  /// error handler did not throw.
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

  /// Read some data from the stream. Returns the number of bytes read. Throws
  /// an exception on failure.
  template <typename Mutable_Buffers>
  std::size_t read_some(const Mutable_Buffers& buffers)
  {
    return stream_impl_.read_some(buffers);
  }

  /// Read some data from the stream. Returns the number of bytes read or 0 if
  /// an error occurred and the error handler did not throw an exception.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t read_some(const Mutable_Buffers& buffers,
      Error_Handler error_handler)
  {
    return stream_impl_.read_some(buffers, error_handler);
  }

  /// Start an asynchronous read. The buffer into which the data will be read
  /// must be valid for the lifetime of the asynchronous operation.
  template <typename Mutable_Buffers, typename Handler>
  void async_read_some(const Mutable_Buffers& buffers, Handler handler)
  {
    stream_impl_.async_read_some(buffers, handler);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read.
  /// Throws an exception on failure.
  template <typename Mutable_Buffers>
  std::size_t peek(const Mutable_Buffers& buffers)
  {
    return stream_impl_.peek(buffers);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read,
  /// or 0 if an error occurred and the error handler did not throw.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t peek(const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    return stream_impl_.peek(buffers, error_handler);
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
  // The buffered write stream.
  typedef buffered_write_stream<Stream> write_stream_type;
  write_stream_type inner_stream_impl_;

  // The buffered read stream.
  typedef buffered_read_stream<write_stream_type&> read_stream_type;
  read_stream_type stream_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_STREAM_HPP
