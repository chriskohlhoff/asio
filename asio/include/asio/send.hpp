//
// send.hpp
// ~~~~~~~~
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
 * @param s The stream on which the data is to be sent.
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
template <typename Stream>
size_t send(Stream& s, const void* data, size_t length)
{
  return s.send(data, length);
}

/// Write some data to a stream.
/**
 * This function is used to send data on a stream. The function call will block
 * until the data has been sent successfully or an error occurs.
 *
 * @param s The stream on which the data is to be sent.
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
template <typename Stream, typename Error_Handler>
size_t send(Stream& s, const void* data, size_t length,
    Error_Handler error_handler)
{
  return s.send(data, length, error_handler);
}

/// Start an asynchronous send.
/**
 * This function is used to asynchronously send data on a stream. The function
 * call always returns immediately.
 *
 * @param s The stream on which the data is to be sent.
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
template <typename Stream, typename Handler>
void async_send(Stream& s, const void* data, size_t length, Handler handler)
{
  s.async_send(data, length, handler);
}

/// Write all of the given data to the stream before returning.
/**
 * This function is used to send an exact number of bytes of data on a stream.
 * The function call will block until the specified number of bytes has been
 * sent successfully or an error occurs.
 *
 * @param s The stream on which the data is to be sent.
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
template <typename Stream>
size_t send_n(Stream& s, const void* data, size_t length,
    size_t* total_bytes_sent = 0)
{
  int bytes_sent = 0;
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
 * @param s The stream on which the data is to be sent.
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
template <typename Stream, typename Error_Handler>
size_t send_n(Stream& s, const void* data, size_t length,
    size_t* total_bytes_sent, Error_Handler error_handler)
{
  int bytes_sent = 0;
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
  template <typename Stream, typename Handler>
  class send_n_handler
  {
  public:
    send_n_handler(Stream& stream, const void* data, size_t length,
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
        async_send(stream_, static_cast<const char*>(data_) + total_sent_,
            length_ - total_sent_, *this);
      }
    }

  private:
    Stream& stream_;
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
 * @param s The stream on which the data is to be sent.
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
template <typename Stream, typename Handler>
void async_send_n(Stream& s, const void* data, size_t length, Handler handler)
{
  async_send(s, data, length,
      detail::send_n_handler<Stream, Handler>(s, data, length, handler));
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SEND_HPP
