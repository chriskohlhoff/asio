//
// buffered_read_stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <cstddef>
#include <cstring>
#include <boost/config.hpp>
#include <boost/type_traits.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_read_stream_fwd.hpp"
#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_resize_guard.hpp"
#include "asio/detail/buffered_stream_storage.hpp"
#include "asio/detail/noncopyable.hpp"

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
 * Async_Object, Async_Read_Stream, Async_Write_Stream, Error_Source, Stream,
 * Sync_Read_Stream, Sync_Write_Stream.
 */
template <typename Stream>
class buffered_read_stream
  : private noncopyable
{
public:
  /// The type of the next layer.
  typedef typename boost::remove_reference<Stream>::type next_layer_type;

  /// The type of the lowest layer.
  typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

  /// The type used for reporting errors.
  typedef typename next_layer_type::error_type error_type;

#if defined(GENERATING_DOCUMENTATION)
  /// The default buffer size.
  static const std::size_t default_buffer_size = implementation_defined;
#else
  BOOST_STATIC_CONSTANT(std::size_t, default_buffer_size = 1024);
#endif

  /// Construct, passing the specified argument to initialise the next layer.
  template <typename Arg>
  explicit buffered_read_stream(Arg& a)
    : next_layer_(a),
      storage_(default_buffer_size)
  {
  }

  /// Construct, passing the specified argument to initialise the next layer.
  template <typename Arg>
  buffered_read_stream(Arg& a, std::size_t buffer_size)
    : next_layer_(a),
      storage_(buffer_size)
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

  /// Get the io_service associated with the object.
  asio::io_service& io_service()
  {
    return next_layer_.io_service();
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

  /// Write the given data to the stream. Returns the number of bytes written.
  /// Throws an exception on failure.
  template <typename Const_Buffers>
  std::size_t write_some(const Const_Buffers& buffers)
  {
    return next_layer_.write_some(buffers);
  }

  /// Write the given data to the stream. Returns the number of bytes written,
  /// or 0 if an error occurred and the error handler did not throw.
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t write_some(const Const_Buffers& buffers,
      Error_Handler error_handler)
  {
    return next_layer_.write_some(buffers, error_handler);
  }

  /// Start an asynchronous write. The data being written must be valid for the
  /// lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Handler>
  void async_write_some(const Const_Buffers& buffers, Handler handler)
  {
    next_layer_.async_write_some(buffers, handler);
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation. Throws an exception on failure.
  std::size_t fill()
  {
    detail::buffer_resize_guard<detail::buffered_stream_storage>
      resize_guard(storage_);
    std::size_t previous_size = storage_.size();
    storage_.resize(storage_.capacity());
    storage_.resize(previous_size + next_layer_.read_some(buffer(
            storage_.data() + previous_size,
            storage_.size() - previous_size)));
    resize_guard.commit();
    return storage_.size() - previous_size;
  }

  /// Fill the buffer with some data. Returns the number of bytes placed in the
  /// buffer as a result of the operation, or 0 if an error occurred and the
  /// error handler did not throw.
  template <typename Error_Handler>
  std::size_t fill(Error_Handler error_handler)
  {
    detail::buffer_resize_guard<detail::buffered_stream_storage>
      resize_guard(storage_);
    std::size_t previous_size = storage_.size();
    storage_.resize(storage_.capacity());
    storage_.resize(previous_size + next_layer_.read_some(buffer(
            storage_.data() + previous_size,
            storage_.size() - previous_size),
          error_handler));
    resize_guard.commit();
    return storage_.size() - previous_size;
  }

  template <typename Handler>
  class fill_handler
  {
  public:
    fill_handler(asio::io_service& io_service,
        detail::buffered_stream_storage& storage,
        std::size_t previous_size, Handler handler)
      : io_service_(io_service),
        storage_(storage),
        previous_size_(previous_size),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, std::size_t bytes_transferred)
    {
      storage_.resize(previous_size_ + bytes_transferred);
      io_service_.dispatch(detail::bind_handler(
            handler_, e, bytes_transferred));
    }

  private:
    asio::io_service& io_service_;
    detail::buffered_stream_storage& storage_;
    std::size_t previous_size_;
    Handler handler_;
  };

  /// Start an asynchronous fill.
  template <typename Handler>
  void async_fill(Handler handler)
  {
    std::size_t previous_size = storage_.size();
    storage_.resize(storage_.capacity());
    next_layer_.async_read_some(
        buffer(
          storage_.data() + previous_size,
          storage_.size() - previous_size),
        fill_handler<Handler>(io_service(), storage_, previous_size, handler));
  }

  /// Read some data from the stream. Returns the number of bytes read. Throws
  /// an exception on failure.
  template <typename Mutable_Buffers>
  std::size_t read_some(const Mutable_Buffers& buffers)
  {
    if (storage_.empty())
      fill();
    return copy(buffers);
  }

  /// Read some data from the stream. Returns the number of bytes read or 0 if
  /// an error occurred and the error handler did not throw an exception.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t read_some(const Mutable_Buffers& buffers,
      Error_Handler error_handler)
  {
    if (storage_.empty() && !fill(error_handler))
      return 0;
    return copy(buffers);
  }

  template <typename Mutable_Buffers, typename Handler>
  class read_some_handler
  {
  public:
    read_some_handler(asio::io_service& io_service,
        detail::buffered_stream_storage& storage,
        const Mutable_Buffers& buffers, Handler handler)
      : io_service_(io_service),
        storage_(storage),
        buffers_(buffers),
        handler_(handler)
    {
    }

    void operator()(const error_type& e, std::size_t)
    {
      if (e || storage_.empty())
      {
        std::size_t length = 0;
        io_service_.dispatch(detail::bind_handler(handler_, e, length));
      }
      else
      {
        using namespace std; // For memcpy.

        std::size_t bytes_avail = storage_.size();
        std::size_t bytes_copied = 0;

        typename Mutable_Buffers::const_iterator iter = buffers_.begin();
        typename Mutable_Buffers::const_iterator end = buffers_.end();
        for (; iter != end && bytes_avail > 0; ++iter)
        {
          std::size_t max_length = buffer_size(*iter);
          std::size_t length = (max_length < bytes_avail)
            ? max_length : bytes_avail;
          memcpy(buffer_cast<void*>(*iter),
              storage_.data() + bytes_copied, length);
          bytes_copied += length;
          bytes_avail -= length;
        }

        storage_.consume(bytes_copied);
        io_service_.dispatch(detail::bind_handler(handler_, e, bytes_copied));
      }
    }

  private:
    asio::io_service& io_service_;
    detail::buffered_stream_storage& storage_;
    Mutable_Buffers buffers_;
    Handler handler_;
  };

  /// Start an asynchronous read. The buffer into which the data will be read
  /// must be valid for the lifetime of the asynchronous operation.
  template <typename Mutable_Buffers, typename Handler>
  void async_read_some(const Mutable_Buffers& buffers, Handler handler)
  {
    if (storage_.empty())
    {
      async_fill(read_some_handler<Mutable_Buffers, Handler>(
            io_service(), storage_, buffers, handler));
    }
    else
    {
      std::size_t length = copy(buffers);
      io_service().post(detail::bind_handler(handler, 0, length));
    }
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read.
  /// Throws an exception on failure.
  template <typename Mutable_Buffers>
  std::size_t peek(const Mutable_Buffers& buffers)
  {
    if (storage_.empty())
      fill();
    return peek_copy(buffers);
  }

  /// Peek at the incoming data on the stream. Returns the number of bytes read,
  /// or 0 if an error occurred and the error handler did not throw.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t peek(const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    if (storage_.empty() && !fill(error_handler))
      return 0;
    return peek_copy(buffers);
  }

  /// Determine the amount of data that may be read without blocking.
  std::size_t in_avail()
  {
    return storage_.size();
  }

  /// Determine the amount of data that may be read without blocking.
  template <typename Error_Handler>
  std::size_t in_avail(Error_Handler error_handler)
  {
    return storage_.size();
  }

private:
  /// Copy data out of the internal buffer to the specified target buffer.
  /// Returns the number of bytes copied.
  template <typename Mutable_Buffers>
  std::size_t copy(const Mutable_Buffers& buffers)
  {
    using namespace std; // For memcpy.

    std::size_t bytes_avail = storage_.size();
    std::size_t bytes_copied = 0;

    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    for (; iter != end && bytes_avail > 0; ++iter)
    {
      std::size_t max_length = buffer_size(*iter);
      std::size_t length = (max_length < bytes_avail)
        ? max_length : bytes_avail;
      memcpy(buffer_cast<void*>(*iter), storage_.data() + bytes_copied, length);
      bytes_copied += length;
      bytes_avail -= length;
    }

    storage_.consume(bytes_copied);
    return bytes_copied;
  }

  /// Copy data from the internal buffer to the specified target buffer, without
  /// removing the data from the internal buffer. Returns the number of bytes
  /// copied.
  template <typename Mutable_Buffers>
  std::size_t peek_copy(const Mutable_Buffers& buffers)
  {
    using namespace std; // For memcpy.

    std::size_t bytes_avail = storage_.size();
    std::size_t bytes_copied = 0;

    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    for (; iter != end && bytes_avail > 0; ++iter)
    {
      std::size_t max_length = buffer_size(*iter);
      std::size_t length = (max_length < bytes_avail)
        ? max_length : bytes_avail;
      memcpy(buffer_cast<void*>(*iter), storage_.data() + bytes_copied, length);
      bytes_copied += length;
      bytes_avail -= length;
    }

    return bytes_copied;
  }

  /// The next layer.
  Stream next_layer_;

  // The data in the buffer.
  detail::buffered_stream_storage storage_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_READ_STREAM_HPP
