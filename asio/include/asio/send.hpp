//
// send.hpp
// ~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SEND_HPP
#define ASIO_SEND_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/bind_handler.hpp"

namespace asio {

/// Write some data to a stream.
/**
 * This function is used to send data on a stream. The function call will block
 * until the data has been sent successfully or an error occurs.
 *
 * @param s The stream on which the data is to be sent. The type must support
 * the Sync_Send_Stream concept.
 *
 * @param data The data to be sent on the stream.
 *
 * @param length The size of the data to be sent, in bytes.
 *
 * @returns The number of bytes sent, or 0 if end-of-file was reached or the
 * connection was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's send operation.
 *
 * @note The send operation may not transmit all of the data to the peer.
 * Consider using the asio::send_n() function if you need to ensure that all
 * data is sent before the blocking operation completes.
 */
template <typename Sync_Send_Stream>
inline size_t send(Sync_Send_Stream& s, const void* data, size_t length)
{
  return s.send(data, length);
}

/// Write some data to a stream.
/**
 * This function is used to send data on a stream. The function call will block
 * until the data has been sent successfully or an error occurs.
 *
 * @param s The stream on which the data is to be sent. The type must support
 * the Sync_Send_Stream concept.
 *
 * @param data The data to be sent on the stream.
 *
 * @param length The size of the data to be sent, in bytes.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's send operation)
 * ); @endcode
 *
 * @returns The number of bytes sent, or 0 if end-of-file was reached or the
 * connection was closed cleanly.
 *
 * @note The send operation may not transmit all of the data to the peer.
 * Consider using the asio::send_n() function if you need to ensure that all
 * data is sent before the blocking operation completes.
 */
template <typename Sync_Send_Stream, typename Error_Handler>
inline size_t send(Sync_Send_Stream& s, const void* data, size_t length,
    Error_Handler error_handler)
{
  return s.send(data, length, error_handler);
}

/// Start an asynchronous send.
/**
 * This function is used to asynchronously send data on a stream. The function
 * call always returns immediately.
 *
 * @param s The stream on which the data is to be sent. The type must support
 * the Async_Send_Stream concept.
 *
 * @param data The data to be sent on the stream. Ownership of the data is
 * retained by the caller, which must guarantee that it is valid until the
 * handler is called.
 *
 * @param length The size of the data to be sent, in bytes.
 *
 * @param handler The handler to be called when the send operation completes.
 * Copies will be made of the handler as required. The equivalent function
 * signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error, // Result of operation (the actual type is dependent
 *                       // on the underlying stream's send operation)
 *   size_t bytes_sent   // Number of bytes sent
 * ); @endcode
 *
 * @note The send operation may not transmit all of the data to the peer.
 * Consider using the asio::async_send_n() function if you need to ensure that
 * all data is sent before the asynchronous operation completes.
 */
template <typename Async_Send_Stream, typename Handler>
inline void async_send(Async_Send_Stream& s, const void* data, size_t length,
    Handler handler)
{
  s.async_send(data, length, handler);
}

/// Write all of the given data to the stream before returning.
/**
 * This function is used to send an exact number of bytes of data on a stream.
 * The function call will block until the specified number of bytes has been
 * sent successfully or an error occurs.
 *
 * @param s The stream on which the data is to be sent. The type must support
 * the Sync_Send_Stream concept.
 *
 * @param data The data to be sent on the stream.
 *
 * @param length The size of the data to be sent, in bytes.
 *
 * @param total_bytes_sent An optional output parameter that receives the
 * total number of bytes actually sent.
 *
 * @returns The number of bytes sent on the last send, or 0 if end-of-file was
 * reached or the connection was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's send operation.
 */
template <typename Sync_Send_Stream>
size_t send_n(Sync_Send_Stream& s, const void* data, size_t length,
    size_t* total_bytes_sent = 0)
{
  size_t bytes_sent = 0;
  size_t total_sent = 0;
  while (total_sent < length)
  {
    bytes_sent = send(s, static_cast<const char*>(data) + total_sent,
        length - total_sent);
    if (bytes_sent == 0)
    {
      if (total_bytes_sent)
        *total_bytes_sent = total_sent;
      return bytes_sent;
    }
    total_sent += bytes_sent;
  }
  if (total_bytes_sent)
    *total_bytes_sent = total_sent;
  return bytes_sent;
}

/// Write all of the given data to the stream before returning.
/**
 * This function is used to send an exact number of bytes of data on a stream.
 * The function call will block until the specified number of bytes has been
 * sent successfully or an error occurs.
 *
 * @param s The stream on which the data is to be sent. The type must support
 * the Sync_Send_Stream concept.
 *
 * @param data The data to be sent on the stream.
 *
 * @param length The size of the data to be sent, in bytes.
 *
 * @param total_bytes_sent An optional output parameter that receives the
 * total number of bytes actually sent.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's send operation)
 * ); @endcode
 *
 * @returns The number of bytes sent on the last send, or 0 if end-of-file was
 * reached or the connection was closed cleanly.
 */
template <typename Sync_Send_Stream, typename Error_Handler>
size_t send_n(Sync_Send_Stream& s, const void* data, size_t length,
    size_t* total_bytes_sent, Error_Handler error_handler)
{
  size_t bytes_sent = 0;
  size_t total_sent = 0;
  while (total_sent < length)
  {
    bytes_sent = send(s, static_cast<const char*>(data) + total_sent,
        length - total_sent, error_handler);
    if (bytes_sent == 0)
    {
      if (total_bytes_sent)
        *total_bytes_sent = total_sent;
      return bytes_sent;
    }
    total_sent += bytes_sent;
  }
  if (total_bytes_sent)
    *total_bytes_sent = total_sent;
  return bytes_sent;
}

namespace detail
{
  template <typename Async_Send_Stream, typename Handler>
  class send_n_handler
  {
  public:
    send_n_handler(Async_Send_Stream& stream, const void* data, size_t length,
        Handler handler)
      : stream_(stream),
        data_(data),
        length_(length),
        total_sent_(0),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_sent)
    {
      total_sent_ += bytes_sent;
      if (e || bytes_sent == 0 || total_sent_ == length_)
      {
        stream_.demuxer().dispatch(
            detail::bind_handler(handler_, e, bytes_sent, total_sent_));
      }
      else
      {
        asio::async_send(stream_,
            static_cast<const char*>(data_) + total_sent_,
            length_ - total_sent_, *this);
      }
    }

  private:
    Async_Send_Stream& stream_;
    const void* data_;
    size_t length_;
    size_t total_sent_;
    Handler handler_;
  };
} // namespace detail

/// Start an asynchronous send that will not complete until the specified
/// amount of data has been sent.
/**
 * This function is used to asynchronously send an exact number of bytes of
 * data on a stream. The function call always returns immediately.
 *
 * @param s The stream on which the data is to be sent. The type must support
 * the Async_Send_Stream concept.
 *
 * @param data The data to be sent on the stream. Ownership of the data is
 * retained by the caller, which must guarantee that it is valid until the
 * handler is called.
 *
 * @param length The size of the data to be sent, in bytes.
 *
 * @param handler The handler to be called when the send operation completes.
 * Copies will be made of the handler as required. The equivalent function
 * signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,      // Result of operation (the actual type is
 *                            // dependent on the underlying stream's send
 *                            // operation)
 *   size_t last_bytes_sent,  // Number of bytes sent on last send operation
 *   size_t total_bytes_sent  // Total number of bytes successfully sent
 * ); @endcode
 */
template <typename Async_Send_Stream, typename Handler>
inline void async_send_n(Async_Send_Stream& s, const void* data, size_t length,
    Handler handler)
{
  async_send(s, data, length,
      detail::send_n_handler<Async_Send_Stream, Handler>(s, data, length,
        handler));
}

/// Write at least a specified number of bytes of data to the stream before
/// returning.
/**
 * This function is used to send at least a specified number of bytes of data
 * on a stream. The function call will block until at least that number of
 * bytes has been sent successfully or an error occurs.
 *
 * @param s The stream on which the data is to be sent. The type must support
 * the Sync_Send_Stream concept.
 *
 * @param data The data to be sent on the stream.
 *
 * @param min_length The minimum size of data to be sent, in bytes.
 *
 * @param max_length The maximum size of data to be sent, in bytes.
 *
 * @param total_bytes_sent An optional output parameter that receives the
 * total number of bytes actually sent.
 *
 * @returns The number of bytes sent on the last send, or 0 if end-of-file was
 * reached or the connection was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's send operation.
 */
template <typename Sync_Send_Stream>
size_t send_at_least_n(Sync_Send_Stream& s, const void* data,
    size_t min_length, size_t max_length, size_t* total_bytes_sent = 0)
{
  size_t bytes_sent = 0;
  size_t total_sent = 0;
  if (max_length < min_length)
    min_length = max_length;
  while (total_sent < min_length)
  {
    bytes_sent = send(s, static_cast<const char*>(data) + total_sent,
        max_length - total_sent);
    if (bytes_sent == 0)
    {
      if (total_bytes_sent)
        *total_bytes_sent = total_sent;
      return bytes_sent;
    }
    total_sent += bytes_sent;
  }
  if (total_bytes_sent)
    *total_bytes_sent = total_sent;
  return bytes_sent;
}

/// Write at least a specified number of bytes of data to the stream before
/// returning.
/**
 * This function is used to send at least a specified number of bytes of data
 * on a stream. The function call will block until at least that number of
 * bytes has been sent successfully or an error occurs.
 *
 * @param s The stream on which the data is to be sent. The type must support
 * the Sync_Send_Stream concept.
 *
 * @param data The data to be sent on the stream.
 *
 * @param min_length The minimum size of data to be sent, in bytes.
 *
 * @param max_length The maximum size of data to be sent, in bytes.
 *
 * @param total_bytes_sent An optional output parameter that receives the
 * total number of bytes actually sent.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's send operation)
 * ); @endcode
 *
 * @returns The number of bytes sent on the last send, or 0 if end-of-file was
 * reached or the connection was closed cleanly.
 */
template <typename Sync_Send_Stream, typename Error_Handler>
size_t send_at_least_n(Sync_Send_Stream& s, const void* data,
    size_t min_length, size_t max_length, size_t* total_bytes_sent,
    Error_Handler error_handler)
{
  size_t bytes_sent = 0;
  size_t total_sent = 0;
  if (max_length < min_length)
    min_length = max_length;
  while (total_sent < min_length)
  {
    bytes_sent = send(s, static_cast<const char*>(data) + total_sent,
        max_length - total_sent, error_handler);
    if (bytes_sent == 0)
    {
      if (total_bytes_sent)
        *total_bytes_sent = total_sent;
      return bytes_sent;
    }
    total_sent += bytes_sent;
  }
  if (total_bytes_sent)
    *total_bytes_sent = total_sent;
  return bytes_sent;
}

namespace detail
{
  template <typename Async_Send_Stream, typename Handler>
  class send_at_least_n_handler
  {
  public:
    send_at_least_n_handler(Async_Send_Stream& stream, const void* data,
        size_t min_length, size_t max_length, Handler handler)
      : stream_(stream),
        data_(data),
        min_length_(min_length),
        max_length_(max_length),
        total_sent_(0),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_sent)
    {
      total_sent_ += bytes_sent;
      if (e || bytes_sent == 0 || total_sent_ >= min_length_)
      {
        stream_.demuxer().dispatch(
            detail::bind_handler(handler_, e, bytes_sent, total_sent_));
      }
      else
      {
        asio::async_send(stream_,
            static_cast<const char*>(data_) + total_sent_,
            max_length_ - total_sent_, *this);
      }
    }

  private:
    Async_Send_Stream& stream_;
    const void* data_;
    size_t min_length_;
    size_t max_length_;
    size_t total_sent_;
    Handler handler_;
  };
} // namespace detail

/// Start an asynchronous send that will not complete until at least the
/// specified amount of data has been sent.
/**
 * This function is used to asynchronously send at least a specified number of
 * bytes of data on a stream. The function call always returns immediately.
 *
 * @param s The stream on which the data is to be sent. The type must support
 * the Async_Send_Stream concept.
 *
 * @param data The data to be sent on the stream. Ownership of the data is
 * retained by the caller, which must guarantee that it is valid until the
 * handler is called.
 *
 * @param min_length The minimum size of data to be sent, in bytes.
 *
 * @param max_length The maximum size of data to be sent, in bytes.
 *
 * @param handler The handler to be called when the send operation completes.
 * Copies will be made of the handler as required. The equivalent function
 * signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,      // Result of operation (the actual type is
 *                            // dependent on the underlying stream's send
 *                            // operation)
 *   size_t last_bytes_sent,  // Number of bytes sent on last send operation
 *   size_t total_bytes_sent  // Total number of bytes successfully sent
 * ); @endcode
 */
template <typename Async_Send_Stream, typename Handler>
inline void async_send_at_least_n(Async_Send_Stream& s, const void* data,
    size_t min_length, size_t max_length, Handler handler)
{
  if (max_length < min_length)
    min_length = max_length;
  async_send(s, data, max_length,
      detail::send_at_least_n_handler<Async_Send_Stream, Handler>(s, data,
        min_length, max_length, handler));
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SEND_HPP
