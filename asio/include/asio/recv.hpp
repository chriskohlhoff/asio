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

#include "asio/is_recv_buffered.hpp"
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
inline size_t recv(Stream& s, void* data, size_t max_length)
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
inline size_t recv(Stream& s, void* data, size_t max_length,
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
 *   const Error& error, // Result of operation (the actual type is dependent
 *                       // on the underlying stream's recv operation)
 *   size_t bytes_recvd  // Number of bytes received
 * ); @endcode
 *
 * @note The recv operation may not receive all of the requested number of
 * bytes. Consider using the asio::async_recv_n() function if you need to
 * ensure that the requested amount of data is received before the asynchronous
 * operation completes.
 */
template <typename Stream, typename Handler>
inline void async_recv(Stream& s, void* data, size_t max_length,
    Handler handler)
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
              bytes_recvd, total_recvd_));
      }
      else
      {
        asio::async_recv(stream_, static_cast<char*>(data_) + total_recvd_,
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
 *   size_t last_bytes_recvd,  // Number of bytes received on last recv
 *                             // operation
 *   size_t total_bytes_recvd  // Total number of bytes successfully received
 * ); @endcode
 */
template <typename Stream, typename Handler>
inline void async_recv_n(Stream& s, void* data, size_t length, Handler handler)
{
  async_recv(s, data, length,
      detail::recv_n_handler<Stream, Handler>(s, data, length, handler));
}

/// Read at least the specified amount of data from the stream before
/// returning.
/**
 * This function is used to receive at least a specified number of bytes of
 * data on a stream. The function call will block until at least that number of
 * bytes has been received successfully or an error occurs.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 *
 * @param min_length The minimum size of the data to be received, in bytes.
 *
 * @param max_length The maximum size of the data to be received, in bytes.
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
size_t recv_at_least_n(Stream& s, void* data, size_t min_length,
    size_t max_length, size_t* total_bytes_recvd = 0)
{
  int bytes_recvd = 0;
  size_t total_recvd = 0;
  if (max_length < min_length)
    min_length = max_length;
  while (total_recvd < min_length)
  {
    bytes_recvd = recv(s, static_cast<char*>(data) + total_recvd,
        max_length - total_recvd);
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

/// Read at least the specified amount of data from the stream before
/// returning.
/**
 * This function is used to receive at least a specified number of bytes of
 * data on a stream. The function call will block until at least that number of
 * bytes has been received successfully or an error occurs.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 *
 * @param min_length The minimum size of the data to be received, in bytes.
 *
 * @param max_length The maximum size of the data to be received, in bytes.
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
size_t recv_at_least_n(Stream& s, void* data, size_t min_length,
    size_t max_length, size_t* total_bytes_recvd, Error_Handler error_handler)
{
  int bytes_recvd = 0;
  size_t total_recvd = 0;
  if (max_length < min_length)
    min_length = max_length;
  while (total_recvd < min_length)
  {
    bytes_recvd = recv(s, static_cast<char*>(data) + total_recvd,
        max_length - total_recvd, error_handler);
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
  class recv_at_least_n_handler
  {
  public:
    recv_at_least_n_handler(Stream& stream, void* data, size_t min_length,
        size_t max_length, Handler handler)
      : stream_(stream),
        data_(data),
        min_length_(min_length),
        max_length_(max_length),
        total_recvd_(0),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_recvd)
    {
      total_recvd_ += bytes_recvd;
      if (e || bytes_recvd == 0 || total_recvd_ >= min_length_)
      {
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e,
              bytes_recvd, total_recvd_));
      }
      else
      {
        asio::async_recv(stream_, static_cast<char*>(data_) + total_recvd_,
            max_length_ - total_recvd_, *this);
      }
    }

  private:
    Stream& stream_;
    void* data_;
    size_t min_length_;
    size_t max_length_;
    size_t total_recvd_;
    Handler handler_;
  };
} // namespace detail

/// Start an asynchronous receive that will not complete until at least the
/// specified amount of data has been received.
/**
 * This function is used to asynchronously receive at least a specified number
 * of bytes of data on a stream. The function call always returns immediately.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 * Ownership of the buffer is retained by the caller, which must guarantee
 * that it is valid until the handler is called.
 *
 * @param min_length The minimum size of the data to be received, in bytes.
 *
 * @param max_length The maximum size of the data to be received, in bytes.
 *
 * @param handler The handler to be called when the receive operation
 * completes. Copies will be made of the handler as required. The equivalent
 * function signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,       // Result of operation (the actual type is
 *                             // dependent on the underlying stream's recv
 *                             // operation)
 *   size_t last_bytes_recvd,  // Number of bytes received on last recv
 *                             // operation
 *   size_t total_bytes_recvd  // Total number of bytes successfully received
 * ); @endcode
 */
template <typename Stream, typename Handler>
inline void async_recv_at_least_n(Stream& s, void* data, size_t min_length,
    size_t max_length, Handler handler)
{
  if (max_length < min_length)
    min_length = max_length;
  async_recv(s, data, max_length,
      detail::recv_at_least_n_handler<Stream, Handler>(s, data, min_length,
        max_length, handler));
}

/// Read data from the stream until a delimiter is reached.
/**
 * This function is used to receive data from the stream until a specified
 * delimiter is reached. The function call will block until the delimiter is
 * found or an error occurs.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 *
 * @param max_data_length The maximum size of the data to be received, in
 * bytes.
 *
 * @param delim The pattern marking the end of the data to receive.
 *
 * @param delim_length The size of the delimiter, in bytes. If the delimiter
 * length is 0 then this function behaves the same as asio::recv.
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
size_t recv_until(Stream& s, void* data, size_t max_data_length,
    const void* delim, size_t delim_length, size_t* total_bytes_recvd = 0)
{
  if (delim_length == 0)
    return recv(s, data, max_data_length);

  unsigned char* data_chars = static_cast<unsigned char*>(data);
  const unsigned char* delim_chars = static_cast<const unsigned char*>(delim);
  size_t data_pos = 0;
  size_t delim_pos = 0;
  size_t last_bytes_recvd = 0;

  while (data_pos < max_data_length && delim_pos < delim_length)
  {
    last_bytes_recvd = recv(s, data_chars + data_pos, 1);
    if (last_bytes_recvd == 0)
    {
      if (total_bytes_recvd)
        *total_bytes_recvd = data_pos;
      return 0;
    }

    if (data_chars[data_pos] == delim_chars[delim_pos])
      ++delim_pos;
    else if (data_chars[data_pos] == delim_chars[0])
      delim_pos = 1;
    else
      delim_pos = 0;

    ++data_pos;
  }

  if (total_bytes_recvd)
    *total_bytes_recvd = data_pos;
  return last_bytes_recvd;
}

namespace detail
{
  template <typename Stream, typename Handler>
  class recv_until_no_delim_handler
  {
  public:
    recv_until_no_delim_handler(Stream& stream, Handler handler)
      : stream_(stream),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, size_t bytes_recvd)
    {
      stream_.demuxer().dispatch(
          detail::bind_handler(handler_, e, bytes_recvd, bytes_recvd));
    }

  private:
    Stream& stream_;
    Handler handler_;
  };

  template <bool Is_Recv_Buffered>
  struct async_recv_until_impl;

  // Non-buffered implementation of async_recv_until.
  template <>
  struct async_recv_until_impl<false>
  {
    template <typename Stream, typename Handler>
    class recv_until_handler
    {
    public:
      recv_until_handler(Stream& stream, void* data, size_t max_data_length,
          const void* delim, size_t delim_length, Handler handler)
        : stream_(stream),
          data_(static_cast<unsigned char*>(data)),
          max_data_length_(max_data_length),
          delim_(static_cast<const unsigned char*>(delim)),
          delim_length_(delim_length),
          data_pos_(0),
          delim_pos_(0),
          handler_(handler)
      {
      }

      template <typename Error>
      void operator()(const Error& e, size_t bytes_recvd)
      {
        if (e || bytes_recvd == 0)
        {
          stream_.demuxer().dispatch(detail::bind_handler(handler_, e,
                bytes_recvd, data_pos_));
        }
        else
        {
          if (data_[data_pos_] == delim_[delim_pos_])
            ++delim_pos_;
          else if (data_[data_pos_] == delim_[0])
            delim_pos_ = 1;
          else
            delim_pos_ = 0;

          ++data_pos_;

          if (data_pos_ < max_data_length_ && delim_pos_ < delim_length_)
          {
            asio::async_recv(stream_, data_ + data_pos_, 1, *this);
          }
          else
          {
            stream_.demuxer().dispatch(detail::bind_handler(handler_, e,
                  bytes_recvd, data_pos_));
          }
        }
      }

    private:
      Stream& stream_;
      unsigned char* data_;
      size_t max_data_length_;
      const unsigned char* delim_;
      size_t delim_length_;
      size_t data_pos_;
      size_t delim_pos_;
      Handler handler_;
    };

    template <typename Stream, typename Handler>
    static void async_recv_until(Stream& s, void* data, size_t max_data_length,
        const void* delim, size_t delim_length, Handler handler)
    {
      if (delim_length == 0)
      {
        async_recv(s, data, max_data_length,
            recv_until_no_delim_handler<Stream, Handler>(s, handler));
      }
      else
      {
        async_recv(s, data, 1,
            recv_until_handler<Stream, Handler>(s, data, max_data_length,
              delim, delim_length, handler));
      }
    }
  };

  // Buffered implementation of async_recv_until.
  template <>
  struct async_recv_until_impl<true>
  {
    template <typename Stream, typename Handler>
    class recv_until_handler
    {
    public:
      recv_until_handler(Stream& stream, void* data, size_t max_data_length,
          const void* delim, size_t delim_length, Handler handler)
        : stream_(stream),
          data_(static_cast<unsigned char*>(data)),
          max_data_length_(max_data_length),
          delim_(static_cast<const unsigned char*>(delim)),
          delim_length_(delim_length),
          data_pos_(0),
          delim_pos_(0),
          handler_(handler)
      {
      }

      template <typename Error>
      void operator()(const Error& e, size_t bytes_recvd)
      {
        if (e || bytes_recvd == 0)
        {
          stream_.demuxer().dispatch(detail::bind_handler(handler_, e,
                bytes_recvd, data_pos_));
        }
        else
        {
          while (!stream_.recv_buffer().empty())
          {
            data_[data_pos_] = stream_.recv_buffer().front();
            stream_.recv_buffer().pop();

            if (data_[data_pos_] == delim_[delim_pos_])
              ++delim_pos_;
            else if (data_[data_pos_] == delim_[0])
              delim_pos_ = 1;
            else
              delim_pos_ = 0;

            ++data_pos_;

            if (data_pos_ == max_data_length_ || delim_pos_ == delim_length_)
            {
              stream_.demuxer().dispatch(detail::bind_handler(handler_, e, 1,
                    data_pos_));
              return;
            }
          }

          stream_.async_fill(*this);
        }
      }

    private:
      Stream& stream_;
      unsigned char* data_;
      size_t max_data_length_;
      const unsigned char* delim_;
      size_t delim_length_;
      size_t data_pos_;
      size_t delim_pos_;
      Handler handler_;
    };

    template <typename Stream, typename Handler>
    static void async_recv_until(Stream& s, void* data, size_t max_data_length,
        const void* delim, size_t delim_length, Handler handler)
    {
      if (delim_length == 0)
      {
        async_recv(s, data, max_data_length,
            recv_until_no_delim_handler<Stream, Handler>(s, handler));
      }
      else if (s.recv_buffer().empty())
      {
        s.async_fill(recv_until_handler<Stream, Handler>(s, data,
              max_data_length, delim, delim_length, handler));
      }
      else
      {
        s.demuxer().post(detail::bind_handler(
              recv_until_handler<Stream, Handler>(s, data, max_data_length,
                delim, delim_length, handler), 0, 1));
      }
    }
  };
} // namespace detail

/// Start an asynchronous receive that will not complete until a delimiter is
/// reached.
/**
 * This function is used to asynchronously receive data from the stream until a
 * specified delimiter is reached. The function call always returns
 * immediately.
 *
 * @param s The stream on which the data is to be received.
 *
 * @param data The buffer into which the received data will be written.
 * Ownership of the buffer is retained by the caller, which must guarantee
 * that it is valid until the handler is called.
 *
 * @param max_data_length The maximum size of the data to be received, in
 * bytes.
 *
 * @param delim The pattern marking the end of the data to receive. Ownership
 * of the pattern is retained by the caller, which must guarantee that it is
 * valid until the handler is called.
 *
 * @param delim_length The size of the delimiter, in bytes. If the delimiter
 * length is 0 then this function behaves the same as asio::recv.
 *
 * @param handler The handler to be called when the receive operation
 * completes. Copies will be made of the handler as required. The equivalent
 * function signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,       // Result of operation (the actual type is
 *                             // dependent on the underlying stream's recv
 *                             // operation)
 *   size_t last_bytes_recvd,  // Number of bytes received on last recv
 *                             // operation
 *   size_t total_bytes_recvd  // Total number of bytes successfully received
 * ); @endcode
 */
template <typename Stream, typename Handler>
inline void async_recv_until(Stream& s, void* data, size_t max_data_length,
    const void* delim, size_t delim_length, Handler handler)
{
  enum { is_recv_buffered = is_recv_buffered<Stream>::value };
  detail::async_recv_until_impl<is_recv_buffered>::async_recv_until(s, data,
      max_data_length, delim, delim_length, handler);
}

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_RECV_HPP
