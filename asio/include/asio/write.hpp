//
// write.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WRITE_HPP
#define ASIO_WRITE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/bind_handler.hpp"

namespace asio {

/// Write some data to a stream.
/**
 * This function is used to write data to a stream. The function call will block
 * until the data has been written successfully or an error occurs.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param data The data to be written to the stream.
 *
 * @param length The size of the data to be written, in bytes.
 *
 * @returns The number of bytes written, or 0 if the stream was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's write operation.
 *
 * @note The write operation may not write all of the data to the stream.
 * Consider using the asio::write_n() function if you need to ensure that all
 * data is written before the blocking operation completes.
 */
template <typename Sync_Write_Stream>
inline size_t write(Sync_Write_Stream& s, const void* data, size_t length)
{
  return s.write(data, length);
}

/// Write some data to a stream.
/**
 * This function is used to write data to a stream. The function call will block
 * until the data has been written successfully or an error occurs.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param data The data to be written to the stream.
 *
 * @param length The size of the data to be written, in bytes.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's write operation)
 * ); @endcode
 *
 * @returns The number of bytes written, or 0 if the stream was closed cleanly.
 *
 * @note The write operation may not write all of the data to the stream.
 * Consider using the asio::write_n() function if you need to ensure that all
 * data is written before the blocking operation completes.
 */
template <typename Sync_Write_Stream, typename Error_Handler>
inline size_t write(Sync_Write_Stream& s, const void* data, size_t length,
    Error_Handler error_handler)
{
  return s.write(data, length, error_handler);
}

/// Start an asynchronous write.
/**
 * This function is used to asynchronously write data to a stream. The function
 * call always returns immediately.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Async_Write_Stream concept.
 *
 * @param data The data to be written to the stream. Ownership of the data is
 * retained by the caller, which must guarantee that it is valid until the
 * handler is called.
 *
 * @param length The size of the data to be written, in bytes.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The equivalent function
 * signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,      // Result of operation (the actual type is
 *                            // dependent on the underlying stream's write
 *                            // operation)
 *   size_t bytes_transferred // Number of bytes written
 * ); @endcode
 *
 * @note The write operation may not write all of the data to the stream.
 * Consider using the asio::async_write_n() function if you need to ensure that
 * all data is written before the asynchronous operation completes.
 */
template <typename Async_Write_Stream, typename Handler>
inline void async_write(Async_Write_Stream& s, const void* data, size_t length,
    Handler handler)
{
  s.async_write(data, length, handler);
}

/// Write all of the given data to the stream before returning.
/**
 * This function is used to write an exact number of bytes of data to a stream.
 * The function call will block until the specified number of bytes has been
 * written successfully or an error occurs.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param data The data to be written to the stream.
 *
 * @param length The size of the data to be written, in bytes.
 *
 * @param total_bytes_transferred An optional output parameter that receives the
 * total number of bytes actually written.
 *
 * @returns The number of bytes written on the last write, or 0 if the stream
 * was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's write operation.
 */
template <typename Sync_Write_Stream>
size_t write_n(Sync_Write_Stream& s, const void* data, size_t length,
    size_t* total_bytes_transferred = 0)
{
  size_t bytes_transferred = 0;
  size_t total_transferred = 0;
  while (total_transferred < length)
  {
    bytes_transferred = write(s,
        static_cast<const char*>(data) + total_transferred,
        length - total_transferred);
    if (bytes_transferred == 0)
    {
      if (total_bytes_transferred)
        *total_bytes_transferred = total_transferred;
      return bytes_transferred;
    }
    total_transferred += bytes_transferred;
  }
  if (total_bytes_transferred)
    *total_bytes_transferred = total_transferred;
  return bytes_transferred;
}

/// Write all of the given data to the stream before returning.
/**
 * This function is used to write an exact number of bytes of data to a stream.
 * The function call will block until the specified number of bytes has been
 * written successfully or an error occurs.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param data The data to be written to the stream.
 *
 * @param length The size of the data to be written, in bytes.
 *
 * @param total_bytes_transferred An optional output parameter that receives the
 * total number of bytes actually written.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's write operation)
 * ); @endcode
 *
 * @returns The number of bytes written on the last write, or 0 if the stream
 * was closed cleanly.
 */
template <typename Sync_Write_Stream, typename Error_Handler>
size_t write_n(Sync_Write_Stream& s, const void* data, size_t length,
    size_t* total_bytes_transferred, Error_Handler error_handler)
{
  size_t bytes_transferred = 0;
  size_t total_transferred = 0;
  while (total_transferred < length)
  {
    bytes_transferred = write(s,
        static_cast<const char*>(data) + total_transferred,
        length - total_transferred, error_handler);
    if (bytes_transferred == 0)
    {
      if (total_bytes_transferred)
        *total_bytes_transferred = total_transferred;
      return bytes_transferred;
    }
    total_transferred += bytes_transferred;
  }
  if (total_bytes_transferred)
    *total_bytes_transferred = total_transferred;
  return bytes_transferred;
}

namespace detail
{
  template <typename Async_Write_Stream, typename Handler>
  class write_n_handler
  {
  public:
    write_n_handler(Async_Write_Stream& stream, const void* data, size_t length,
        Handler handler)
      : stream_(stream),
        data_(data),
        length_(length),
        total_transferred_(0),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_transferred)
    {
      total_transferred_ += bytes_transferred;
      if (e || bytes_transferred == 0 || total_transferred_ == length_)
      {
        stream_.demuxer().dispatch(
            detail::bind_handler(handler_, e,
              bytes_transferred, total_transferred_));
      }
      else
      {
        asio::async_write(stream_,
            static_cast<const char*>(data_) + total_transferred_,
            length_ - total_transferred_, *this);
      }
    }

  private:
    Async_Write_Stream& stream_;
    const void* data_;
    size_t length_;
    size_t total_transferred_;
    Handler handler_;
  };
} // namespace detail

/// Start an asynchronous write that will not complete until the specified
/// amount of data has been written.
/**
 * This function is used to asynchronously write an exact number of bytes of
 * data to a stream. The function call always returns immediately.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Async_Write_Stream concept.
 *
 * @param data The data to be written to the stream. Ownership of the data is
 * retained by the caller, which must guarantee that it is valid until the
 * handler is called.
 *
 * @param length The size of the data to be written, in bytes.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The equivalent function
 * signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,            // Result of operation (the actual type is
 *                                  // dependent on the underlying stream's
 *                                  // write operation)
 *   size_t last_bytes_transferred, // Number of bytes written on last write
 *                                  // operation
 *   size_t total_bytes_transferred // Total number of bytes successfully
 *                                  // written
 * ); @endcode
 */
template <typename Async_Write_Stream, typename Handler>
inline void async_write_n(Async_Write_Stream& s, const void* data,
    size_t length, Handler handler)
{
  async_write(s, data, length,
      detail::write_n_handler<Async_Write_Stream, Handler>(s, data, length,
        handler));
}

/// Write at least a specified number of bytes of data to the stream before
/// returning.
/**
 * This function is used to write at least a specified number of bytes of data
 * to a stream. The function call will block until at least that number of
 * bytes has been written successfully or an error occurs.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param data The data to be written to the stream.
 *
 * @param min_length The minimum size of data to be written, in bytes.
 *
 * @param max_length The maximum size of data to be written, in bytes.
 *
 * @param total_bytes_transferred An optional output parameter that receives the
 * total number of bytes actually written.
 *
 * @returns The number of bytes written on the last write, or 0 if the stream
 * was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's write operation.
 */
template <typename Sync_Write_Stream>
size_t write_at_least_n(Sync_Write_Stream& s, const void* data,
    size_t min_length, size_t max_length, size_t* total_bytes_transferred = 0)
{
  size_t bytes_transferred = 0;
  size_t total_transferred = 0;
  if (max_length < min_length)
    min_length = max_length;
  while (total_transferred < min_length)
  {
    bytes_transferred = write(s,
        static_cast<const char*>(data) + total_transferred,
        max_length - total_transferred);
    if (bytes_transferred == 0)
    {
      if (total_bytes_transferred)
        *total_bytes_transferred = total_transferred;
      return bytes_transferred;
    }
    total_transferred += bytes_transferred;
  }
  if (total_bytes_transferred)
    *total_bytes_transferred = total_transferred;
  return bytes_transferred;
}

/// Write at least a specified number of bytes of data to the stream before
/// returning.
/**
 * This function is used to write at least a specified number of bytes of data
 * to a stream. The function call will block until at least that number of
 * bytes has been written successfully or an error occurs.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param data The data to be written to the stream.
 *
 * @param min_length The minimum size of data to be written, in bytes.
 *
 * @param max_length The maximum size of data to be written, in bytes.
 *
 * @param total_bytes_transferred An optional output parameter that receives the
 * total number of bytes actually written.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's write operation)
 * ); @endcode
 *
 * @returns The number of bytes written on the last write, or 0 if the stream
 * was closed cleanly.
 */
template <typename Sync_Write_Stream, typename Error_Handler>
size_t write_at_least_n(Sync_Write_Stream& s, const void* data,
    size_t min_length, size_t max_length, size_t* total_bytes_transferred,
    Error_Handler error_handler)
{
  size_t bytes_transferred = 0;
  size_t total_transferred = 0;
  if (max_length < min_length)
    min_length = max_length;
  while (total_transferred < min_length)
  {
    bytes_transferred = write(s,
        static_cast<const char*>(data) + total_transferred,
        max_length - total_transferred, error_handler);
    if (bytes_transferred == 0)
    {
      if (total_bytes_transferred)
        *total_bytes_transferred = total_transferred;
      return bytes_transferred;
    }
    total_transferred += bytes_transferred;
  }
  if (total_bytes_transferred)
    *total_bytes_transferred = total_transferred;
  return bytes_transferred;
}

namespace detail
{
  template <typename Async_Write_Stream, typename Handler>
  class write_at_least_n_handler
  {
  public:
    write_at_least_n_handler(Async_Write_Stream& stream, const void* data,
        size_t min_length, size_t max_length, Handler handler)
      : stream_(stream),
        data_(data),
        min_length_(min_length),
        max_length_(max_length),
        total_transferred_(0),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_transferred)
    {
      total_transferred_ += bytes_transferred;
      if (e || bytes_transferred == 0 || total_transferred_ >= min_length_)
      {
        stream_.demuxer().dispatch(
            detail::bind_handler(handler_, e,
              bytes_transferred, total_transferred_));
      }
      else
      {
        asio::async_write(stream_,
            static_cast<const char*>(data_) + total_transferred_,
            max_length_ - total_transferred_, *this);
      }
    }

  private:
    Async_Write_Stream& stream_;
    const void* data_;
    size_t min_length_;
    size_t max_length_;
    size_t total_transferred_;
    Handler handler_;
  };
} // namespace detail

/// Start an asynchronous write that will not complete until at least the
/// specified amount of data has been written.
/**
 * This function is used to asynchronously write at least a specified number of
 * bytes of data to a stream. The function call always returns immediately.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Async_Write_Stream concept.
 *
 * @param data The data to be written to the stream. Ownership of the data is
 * retained by the caller, which must guarantee that it is valid until the
 * handler is called.
 *
 * @param min_length The minimum size of data to be written, in bytes.
 *
 * @param max_length The maximum size of data to be written, in bytes.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The equivalent function
 * signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,            // Result of operation (the actual type is
 *                                  // dependent on the underlying stream's
 *                                  // write operation)
 *   size_t last_bytes_transferred, // Number of bytes written on last write
 *                                  // operation
 *   size_t total_bytes_transferred // Total number of bytes successfully
 *                                  // written
 * ); @endcode
 */
template <typename Async_Write_Stream, typename Handler>
inline void async_write_at_least_n(Async_Write_Stream& s, const void* data,
    size_t min_length, size_t max_length, Handler handler)
{
  if (max_length < min_length)
    min_length = max_length;
  async_write(s, data, max_length,
      detail::write_at_least_n_handler<Async_Write_Stream, Handler>(s, data,
        min_length, max_length, handler));
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_WRITE_HPP
