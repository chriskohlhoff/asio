//
// recv.hpp
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

#ifndef ASIO_RECV_HPP
#define ASIO_RECV_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <string>
#include <utility>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/bind_handler.hpp"

namespace asio {

/// Read some data from a stream.
/**
 * This function is used to receive data on a stream. The function call will
 * block until data has received successfully or an error occurs.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 *
 * @param max_length The maximum size of the data to be received, in bytes.
 *
 * @returns The number of bytes received, or 0 if end-of-file was reached or
 * the connection was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's recv operation.
 *
 * @note The recv operation may not receive all of the requested number of
 * bytes. Consider using the asio::recv_n() function if you need to ensure that
 * the requested amount of data is received before the blocking operation
 * completes.
 */
template <typename Stream>
size_t recv(Stream& s, void* data, size_t max_length)
{
  return s.recv(data, max_length);
}

/// Read some data from a stream.
/**
 * This function is used to receive data on a stream. The function call will
 * block until data has received successfully or an error occurs.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 *
 * @param max_length The maximum size of the data to be received, in bytes.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's recv operation)
 * ); @endcode
 *
 * @returns The number of bytes received, or 0 if end-of-file was reached or
 * the connection was closed cleanly.
 *
 * @note The recv operation may not receive all of the requested number of
 * bytes. Consider using the asio::recv_n() function if you need to ensure that
 * the requested amount of data is received before the blocking operation
 * completes.
 */
template <typename Stream, typename Error_Handler>
size_t recv(Stream& s, void* data, size_t max_length,
    Error_Handler error_handler)
{
  return s.recv(data, max_length, error_handler);
}

/// Start an asynchronous receive.
/**
 * This function is used to asynchronously receive data on a stream. The
 * function call always returns immediately.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 * Ownership of the buffer is retained by the caller, which must guarantee
 * that it is valid until the handler is called.
 *
 * @param max_length The maximum size of the data to be received, in bytes.
 *
 * @param handler The handler to be called when the receive operation
 * completes. Copies will be made of the handler as required. The equivalent
 * function signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,   // Result of operation (the actual type is dependent
 *                         // on the underlying stream's recv operation)
 *   size_t bytes_received // Number of bytes received
 * ); @endcode
 *
 * @note The recv operation may not receive all of the requested number of
 * bytes. Consider using the asio::async_recv_n() function if you need to
 * ensure that the requested amount of data is received before the asynchronous
 * operation completes.
 */
template <typename Stream, typename Handler>
void async_recv(Stream& s, void* data, size_t max_length, Handler handler)
{
  s.async_recv(data, max_length, handler);
}

/// Read the specified amount of data from the stream before returning.
/**
 * This function is used to receive an exact number of bytes of data on a
 * stream. The function call will block until the specified number of bytes has
 * been received successfully or an error occurs.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 *
 * @param length The size of the data to be received, in bytes.
 *
 * @param total_bytes_recvd An optional output parameter that receives the
 * total number of bytes actually received.
 *
 * @returns The number of bytes received on the last recv, or 0 if end-of-file
 * was reached or the connection was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's recv operation.
 */
template <typename Stream>
size_t recv_n(Stream& s, void* data, size_t length,
    size_t* total_bytes_recvd = 0)
{
  int bytes_recvd = 0;
  size_t total_recvd = 0;
  while (total_recvd < length)
  {
    bytes_recvd = recv(s, static_cast<char*>(data) + total_recvd,
        length - total_recvd);
    if (bytes_recvd == 0)
    {
      if (total_bytes_recvd)
        *total_bytes_recvd = total_recvd;
      return bytes_recvd;
    }
    total_recvd += bytes_recvd;
  }
  if (total_bytes_recvd)
    *total_bytes_recvd = total_recvd;
  return bytes_recvd;
}

/// Read the specified amount of data from the stream before returning.
/**
 * This function is used to receive an exact number of bytes of data on a
 * stream. The function call will block until the specified number of bytes has
 * been received successfully or an error occurs.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 *
 * @param length The size of the data to be received, in bytes.
 *
 * @param total_bytes_recvd An optional output parameter that receives the
 * total number of bytes actually received.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's recv operation)
 * ); @endcode
 *
 * @returns The number of bytes received on the last recv, or 0 if end-of-file
 * was reached or the connection was closed cleanly.
 */
template <typename Stream, typename Error_Handler>
size_t recv_n(Stream& s, void* data, size_t length,
    size_t* total_bytes_recvd, Error_Handler error_handler)
{
  int bytes_recvd = 0;
  size_t total_recvd = 0;
  while (total_recvd < length)
  {
    bytes_recvd = recv(s, static_cast<char*>(data) + total_recvd,
        length - total_recvd, error_handler);
    if (bytes_recvd == 0)
    {
      if (total_bytes_recvd)
        *total_bytes_recvd = total_recvd;
      return bytes_recvd;
    }
    total_recvd += bytes_recvd;
  }
  if (total_bytes_recvd)
    *total_bytes_recvd = total_recvd;
  return bytes_recvd;
}

namespace detail
{
  template <typename Stream, typename Handler>
  class recv_n_handler
  {
  public:
    recv_n_handler(Stream& stream, void* data, size_t length, Handler handler)
      : stream_(stream),
        data_(data),
        length_(length),
        total_recvd_(0),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_recvd)
    {
      total_recvd_ += bytes_recvd;
      if (e || bytes_recvd == 0 || total_recvd_ == length_)
      {
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e,
              total_recvd_, bytes_recvd));
      }
      else
      {
        async_recv(stream_, static_cast<char*>(data_) + total_recvd_,
            length_ - total_recvd_, *this);
      }
    }

  private:
    Stream& stream_;
    void* data_;
    size_t length_;
    size_t total_recvd_;
    Handler handler_;
  };
} // namespace detail

/// Start an asynchronous receive that will not complete until the specified
/// amount of data has been received.
/**
 * This function is used to asynchronously receive an exact number of bytes of
 * data on a stream. The function call always returns immediately.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 * Ownership of the buffer is retained by the caller, which must guarantee
 * that it is valid until the handler is called.
 *
 * @param length The size of the data to be received, in bytes.
 *
 * @param handler The handler to be called when the receive operation
 * completes. Copies will be made of the handler as required. The equivalent
 * function signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,       // Result of operation (the actual type is
 *                             // dependent on the underlying stream's recv
 *                             // operation)
 *   size_t total_bytes_recvd, // Total number of bytes successfully received
 *   size_t last_bytes_recvd   // Number of bytes received on last recv
 *                             // operation
 * ); @endcode
 */
template <typename Stream, typename Handler>
void async_recv_n(Stream& s, void* data, size_t length, Handler handler)
{
  async_recv(s, data, length,
      detail::recv_n_handler<Stream, Handler>(s, data, length, handler));
}

/// Read some data from a stream and decode it.
/**
 * This function is used to receive data on a stream and decode it in a single
 * operation. The function call will block until the decoder function object
 * indicates that it has finished.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param decoder The decoder function object to be called to decode the
 * received data. The function object is assumed to be stateful. That is, it
 * may not be given sufficient data in a single invocation to complete
 * decoding, and is expected to maintain state so that it may resume decoding
 * when the next piece of data is supplied. Copies will be made of the decoder
 * function object as required, however with respect to maintaining state it
 * can rely on the fact that only an up-to-date copy will be used. The
 * equivalent function signature of the handler must be:
 * @code std::pair<bool, const char*> decoder(
 *   const char* begin, // Pointer to the beginning of data to be decoded.
 *   const char* end    // Pointer to one-past-the-end of data to be decoded.
 * ); @endcode
 * The first element of the return value is true if the decoder has finished.
 * The second element is a pointer to the beginning of the unused portion of
 * the data.
 *
 * @param total_bytes_recvd An optional output parameter that receives the
 * total number of bytes actually received.
 *
 * @returns The number of bytes received on the last recv, or 0 if end-of-file
 * was reached or the connection was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's recv operation.
 */
template <typename Buffered_Stream, typename Decoder>
size_t recv_decode(Buffered_Stream& s, Decoder decoder,
    size_t* total_bytes_recvd = 0)
{
  size_t total_recvd = 0;
  for (;;)
  {
    if (s.recv_buffer().empty() && s.fill() == 0)
    {
      if (total_bytes_recvd)
        *total_bytes_recvd = total_recvd;
      return 0;
    }

    std::pair<bool, const char*> result =
      decoder(s.recv_buffer().begin(), s.recv_buffer().end());

    size_t bytes_read = result.second - s.recv_buffer().begin();
    s.recv_buffer().pop(bytes_read);
    total_recvd += bytes_read;

    if (result.first)
    {
      if (total_bytes_recvd)
        *total_bytes_recvd = total_recvd;
      return bytes_read;
    }
  }
}

/// Read some data from a stream and decode it.
/**
 * This function is used to receive data on a stream and decode it in a single
 * operation. The function call will block until the decoder function object
 * indicates that it has finished.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param decoder The decoder function object to be called to decode the
 * received data. The function object is assumed to be stateful. That is, it
 * may not be given sufficient data in a single invocation to complete
 * decoding, and is expected to maintain state so that it may resume decoding
 * when the next piece of data is supplied. Copies will be made of the decoder
 * function object as required, however with respect to maintaining state it
 * can rely on the fact that only an up-to-date copy will be used. The
 * equivalent function signature of the handler must be:
 * @code std::pair<bool, const char*> decoder(
 *   const char* begin, // Pointer to the beginning of data to be decoded.
 *   const char* end    // Pointer to one-past-the-end of data to be decoded.
 * ); @endcode
 * The first element of the return value is true if the decoder has finished.
 * The second element is a pointer to the beginning of the unused portion of
 * the data.
 *
 * @param total_bytes_recvd An optional output parameter that receives the
 * total number of bytes actually received.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's recv operation)
 * ); @endcode
 *
 * @returns The number of bytes received on the last recv, or 0 if end-of-file
 * was reached or the connection was closed cleanly.
 */
template <typename Buffered_Stream, typename Decoder, typename Error_Handler>
size_t recv_decode(Buffered_Stream& s, Decoder decoder,
    size_t* total_bytes_recvd, Error_Handler error_handler)
{
  size_t total_recvd = 0;
  for (;;)
  {
    if (s.recv_buffer().empty() && s.fill(error_handler) == 0)
    {
      if (total_bytes_recvd)
        *total_bytes_recvd = total_recvd;
      return 0;
    }

    std::pair<bool, const char*> result =
      decoder(s.recv_buffer().begin(), s.recv_buffer().end());

    size_t bytes_read = result.second - s.recv_buffer().begin();
    s.recv_buffer().pop(bytes_read);
    total_recvd += bytes_read;

    if (result.first)
    {
      if (total_bytes_recvd)
        *total_bytes_recvd = total_recvd;
      return bytes_read;
    }
  }
}

namespace detail
{
  template <typename Buffered_Stream, typename Decoder, typename Handler>
  class recv_decode_handler
  {
  public:
    recv_decode_handler(Buffered_Stream& stream, Decoder decoder,
        Handler handler)
      : stream_(stream),
        decoder_(decoder),
        total_recvd_(0),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_recvd)
    {
      if (e || bytes_recvd == 0)
      {
        stream_.demuxer().dispatch(
            detail::bind_handler(handler_, e, total_recvd_, bytes_recvd));
      }
      else
      {
        while (!stream_.recv_buffer().empty())
        {
          std::pair<bool, const char*> result =
            decoder_(stream_.recv_buffer().begin(),
                stream_.recv_buffer().end());

          size_t bytes_read = result.second - stream_.recv_buffer().begin();
          stream_.recv_buffer().pop(bytes_read);
          total_recvd_ += bytes_read;

          if (result.first)
          {
            stream_.demuxer().dispatch(
                detail::bind_handler(handler_, 0, total_recvd_, bytes_read));
            return;
          }
        }

        stream_.async_fill(*this);
      }
    }

  private:
    Buffered_Stream& stream_;
    Decoder decoder_;
    size_t total_recvd_;
    Handler handler_;
  };
} // namespace detail

/// Start an asynchronous receive that will not complete until some data has
/// been fully decoded.
/**
 * This function is used to receive data on a stream and decode it in a single
 * asynchronous operation. The function call always returns immediately. The
 * asynchronous operation will complete only when the decoder indicates that it
 * has finished.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param decoder The decoder function object to be called to decode the
 * received data. The function object is assumed to be stateful. That is, it
 * may not be given sufficient data in a single invocation to complete
 * decoding, and is expected to maintain state so that it may resume decoding
 * when the next piece of data is supplied. Copies will be made of the decoder
 * function object as required, however with respect to maintaining state it
 * can rely on the fact that only an up-to-date copy will be used. The
 * equivalent function signature of the handler must be:
 * @code std::pair<bool, const char*> decoder(
 *   const char* begin, // Pointer to the beginning of data to be decoded.
 *   const char* end    // Pointer to one-past-the-end of data to be decoded.
 * ); @endcode
 * The first element of the return value is true if the decoder has finished.
 * The second element is a pointer to the beginning of the unused portion of
 * the data.
 *
 * @param handler The handler to be called when the receive operation
 * completes. Copies will be made of the handler as required. The equivalent
 * function signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,       // Result of operation (the actual type is
 *                             // dependent on the underlying stream's recv
 *                             // operation)
 *   size_t total_bytes_recvd, // Total number of bytes successfully received
 *   size_t last_bytes_recvd   // Number of bytes received on last recv
 *                             // operation
 * ); @endcode
 */
template <typename Buffered_Stream, typename Decoder, typename Handler>
void async_recv_decode(Buffered_Stream& s, Decoder decoder, Handler handler)
{
  while (!s.recv_buffer().empty())
  {
    std::pair<bool, const char*> result =
      decoder(s.recv_buffer().begin(), s.recv_buffer().end());

    size_t bytes_read = result.second - s.recv_buffer().begin();
    s.recv_buffer().pop(bytes_read);

    if (result.first)
    {
      s.demuxer().dispatch(
          detail::bind_handler(handler, 0, bytes_read, bytes_read));
      return;
    }
  }

  s.async_fill(detail::recv_decode_handler<Buffered_Stream, Decoder, Handler>(
        s, decoder, handler));
}

namespace detail
{
  class recv_until_decoder
  {
  public:
    recv_until_decoder(std::string& data, const std::string& delimiter)
      : data_(data),
        delimiter_(delimiter),
        delimiter_length_(delimiter.length()),
        delimiter_pos_(0)
    {
      data_ = "";
    }

    std::pair<bool, const char*> operator()(const char* begin, const char* end)
    {
      const char* p = begin;
      while (p < end)
      {
        char next_char = *p++;
        if (next_char == delimiter_[delimiter_pos_])
        {
          if (++delimiter_pos_ == delimiter_length_)
          {
            data_.append(begin, p - begin);
            return std::make_pair(true, p);
          }
        }
        else
        {
          delimiter_pos_ = 0;
        }
      }
      data_.append(begin, end - begin);
      return std::make_pair(false, end);
    }

  private:
    std::string& data_;
    std::string delimiter_;
    size_t delimiter_length_;
    size_t delimiter_pos_;
  };
}

/// Read data from the stream until a delimiter is reached.
/**
 * This function is used to receive data from the stream into a std::string
 * object until a specified delimiter is reached. The function call will block
 * until the delimiter is found or an error occurs.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The std::string object into which the received data will be
 * written.
 *
 * @param delimiter The pattern marking the end of the data to receive.
 *
 * @param total_bytes_recvd An optional output parameter that receives the
 * total number of bytes actually received.
 *
 * @returns The number of bytes received on the last recv, or 0 if end-of-file
 * was reached or the connection was closed cleanly.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's recv operation.
 */
template <typename Buffered_Stream>
size_t recv_until(Buffered_Stream& s, std::string& data,
    const std::string& delimiter, size_t* total_bytes_recvd = 0)
{
  return recv_decode(s, detail::recv_until_decoder(data, delimiter),
      total_bytes_recvd);
}

/// Read data from the stream until a delimiter is reached.
/**
 * This function is used to receive data from the stream into a std::string
 * object until a specified delimiter is reached. The function call will block
 * until the delimiter is found or an error occurs.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The std::string object into which the received data will be
 * written.
 *
 * @param delimiter The pattern marking the end of the data to receive.
 *
 * @param total_bytes_recvd An optional output parameter that receives the
 * total number of bytes actually received.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's recv operation)
 * ); @endcode
 *
 * @returns The number of bytes received on the last recv, or 0 if end-of-file
 * was reached or the connection was closed cleanly.
 */
template <typename Buffered_Stream, typename Error_Handler>
size_t recv_until(Buffered_Stream& s, std::string& data,
    const std::string& delimiter, size_t* total_bytes_recvd,
    Error_Handler error_handler)
{
  return recv_decode(s, detail::recv_until_decoder(data, delimiter),
      total_bytes_recvd, error_handler);
}

/// Start an asynchronous receive that will not complete until the specified
/// delimiter is encountered.
/**
 * This function is used to asynchronously receive data from a stream until a
 * given delimiter is found. The function call always returns immediately.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The std:::string object into which the received data will be
 * written. Ownership of the object is retained by the caller, which must
 * guarantee that it is valid until the handler is called.
 *
 * @param delimiter The pattern marking the end of the data to receive. Copies
 * will be made of the string as required.
 *
 * @param handler The handler to be called when the receive operation
 * completes. Copies will be made of the handler as required. The equivalent
 * function signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,       // Result of operation (the actual type is
 *                             // dependent on the underlying stream's recv
 *                             // operation)
 *   size_t total_bytes_recvd, // Total number of bytes successfully received
 *   size_t last_bytes_recvd   // Number of bytes received on last recv
 *                             // operation
 * ); @endcode
 */
template <typename Buffered_Stream, typename Handler>
void async_recv_until(Buffered_Stream& s, std::string& data,
    const std::string& delimiter, Handler handler)
{
  async_recv_decode(s, detail::recv_until_decoder(data, delimiter), handler);
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_RECV_HPP
