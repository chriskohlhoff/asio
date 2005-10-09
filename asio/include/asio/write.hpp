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

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/consuming_buffers.hpp"
#include "asio/detail/bind_handler.hpp"

namespace asio {

/**
 * @defgroup write asio::write
 */
/*@{*/

/// Write some data to a stream.
/**
 * This function is used to write data to a stream. The function call will block
 * until the data has been written successfully or an error occurs.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param buffers One or more data buffers to be written to the stream.
 *
 * @returns The number of bytes written.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's write operation.
 *
 * @note The write operation may not write all of the data to the stream.
 * Consider using the @ref write_n function if you need to ensure that all data
 * is written before the blocking operation completes.
 *
 * @par Example:
 * To write a single data buffer use the @ref buffers function as follows:
 * @code asio::write(s, asio::buffers(data, size)); @endcode
 * See the @ref buffers documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename Sync_Write_Stream, typename Const_Buffers>
inline std::size_t write(Sync_Write_Stream& s, const Const_Buffers& buffers)
{
  return s.write(buffers);
}

/// Write some data to a stream.
/**
 * This function is used to write data to a stream. The function call will block
 * until the data has been written successfully or an error occurs.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param buffers One or more data buffers to be written to the stream.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's write operation).
 * ); @endcode
 *
 * @returns The number of bytes written.
 *
 * @note The write operation may not write all of the data to the stream.
 * Consider using the @ref write_n function if you need to ensure that all data
 * is written before the blocking operation completes.
 */
template <typename Sync_Write_Stream, typename Const_Buffers,
    typename Error_Handler>
inline std::size_t write(Sync_Write_Stream& s, const Const_Buffers& buffers,
    Error_Handler error_handler)
{
  return s.write(buffers, error_handler);
}

/*@}*/
/**
 * @defgroup async_write asio::async_write
 */
/*@{*/

/// Start an asynchronous write.
/**
 * This function is used to asynchronously write data to a stream. The function
 * call always returns immediately.
 *
 * @param s The stream to which the data is to be written. The type must support
 * the Async_Write_Stream concept.
 *
 * @param buffers One or more data buffers to be written to the stream.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The equivalent function
 * signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,           // Result of operation (the actual type is
 *                                 // dependent on the underlying stream's write
 *                                 // operation).

 *   std::size_t bytes_transferred // Number of bytes written.
 * ); @endcode
 *
 * @note The write operation may not write all of the data to the stream.
 * Consider using the @ref async_write_n function if you need to ensure that
 * all data is written before the asynchronous operation completes.
 *
 * @par Example:
 * To write a single data buffer use the @ref buffers function as follows:
 * @code asio::async_write(s, asio::buffers(data, size), handler); @endcode
 * See the @ref buffers documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename Async_Write_Stream, typename Const_Buffers, typename Handler>
inline void async_write(Async_Write_Stream& s, const Const_Buffers& buffers,
    Handler handler)
{
  s.async_write(buffers, handler);
}

/*@}*/
/**
 * @defgroup write_n asio::write_n
 */
/*@{*/

/// Write all of the supplied data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * total bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This function is implemented in terms of one or more calls to @ref write.
 * 
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param buffers One or more buffers containing the data to be written.
 *
 * @param total_bytes_transferred An optional output parameter that receives the
 * total number of bytes written from the buffers. If an error occurred this
 * will be less than the sum of the buffer sizes.
 *
 * @returns The number of bytes transferred on the last write.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's write operation.
 *
 * @par Example:
 * To write a single data buffer use the @ref buffers function as follows:
 * @code asio::write_n(s, asio::buffers(data, size)); @endcode
 * See the @ref buffers documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename Sync_Write_Stream, typename Const_Buffers>
std::size_t write_n(Sync_Write_Stream& s, const Const_Buffers& buffers,
    std::size_t* total_bytes_transferred = 0)
{
  consuming_buffers<Const_Buffers> tmp(buffers);
  std::size_t bytes_transferred = 0;
  std::size_t total_transferred = 0;
  while (tmp.begin() != tmp.end())
  {
    bytes_transferred = write(s, tmp);
    if (bytes_transferred == 0)
    {
      if (total_bytes_transferred)
        *total_bytes_transferred = total_transferred;
      return bytes_transferred;
    }
    tmp.consume(bytes_transferred);
    total_transferred += bytes_transferred;
  }
  if (total_bytes_transferred)
    *total_bytes_transferred = total_transferred;
  return bytes_transferred;
}

/// Write all of the supplied data to a stream before returning.
/**
 * This function is used to write a certain number of bytes of data to a stream.
 * The call will block until one of the following conditions is true:
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * total bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This function is implemented in terms of one or more calls to @ref write.
 * 
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param buffers One or more buffers containing the data to be written.
 *
 * @param total_bytes_transferred An optional output parameter that receives the
 * total number of bytes written from the buffers. If an error occurred this
 * will be less than the sum of the buffer sizes.
 *
 * @param error_handler The handler to be called when an error occurs. Copies
 * will be made of the handler as required. The equivalent function signature
 * of the handler must be:
 * @code template <typename Error>
 * void error_handler(
 *   const Error& error // Result of operation (the actual type is dependent on
 *                      // the underlying stream's write operation).
 * ); @endcode
 *
 * @returns The number of bytes transferred on the last write.
 */
template <typename Sync_Write_Stream, typename Const_Buffers,
    typename Error_Handler>
std::size_t write_n(Sync_Write_Stream& s, const Const_Buffers& buffers,
    std::size_t* total_bytes_transferred, Error_Handler error_handler)
{
  consuming_buffers<Const_Buffers> tmp(buffers);
  std::size_t bytes_transferred = 0;
  std::size_t total_transferred = 0;
  while (tmp.begin() != tmp.end())
  {
    bytes_transferred = write(s, tmp, error_handler);
    if (bytes_transferred == 0)
    {
      if (total_bytes_transferred)
        *total_bytes_transferred = total_transferred;
      return bytes_transferred;
    }
    tmp.consume(bytes_transferred);
    total_transferred += bytes_transferred;
  }
  if (total_bytes_transferred)
    *total_bytes_transferred = total_transferred;
  return bytes_transferred;
}

/*@}*/

namespace detail
{
  template <typename Async_Write_Stream, typename Const_Buffers,
      typename Handler>
  class write_n_handler
  {
  public:
    write_n_handler(Async_Write_Stream& stream, const Const_Buffers& buffers,
        Handler handler)
      : stream_(stream),
        buffers_(buffers),
        total_transferred_(0),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, std::size_t bytes_transferred)
    {
      total_transferred_ += bytes_transferred;
      buffers_.consume(bytes_transferred);
      if (e || bytes_transferred == 0 || buffers_.begin() == buffers_.end())
      {
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e,
              bytes_transferred, total_transferred_));
      }
      else
      {
        asio::async_write(stream_, buffers_, *this);
      }
    }

  private:
    Async_Write_Stream& stream_;
    consuming_buffers<Const_Buffers> buffers_;
    std::size_t total_transferred_;
    Handler handler_;
  };
} // namespace detail

/**
 * @defgroup async_write_n asio::async_write_n
 */
/*@{*/

/// Start an asynchronous write of all of the supplied data to a stream.
/**
 * This function is used to asynchronously write a certain number of bytes of
 * data to a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * total bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This function is implemented in terms of one or more calls to @ref
 * async_write.
 * 
 * @param s The stream to which the data is to be written. The type must support
 * the Async_Write_Stream concept.
 *
 * @param buffers One or more buffers containing the data to be written.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The equivalent function
 * signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,                 // Result of operation (the actual type
 *                                       // is dependent on the underlying
 *                                       // stream's write operation).
 *
 *   std::size_t last_bytes_transferred, // Number of bytes transferred on last
 *                                       // write operation.
 *
 *   std::size_t total_bytes_transferred // Total number of bytes successfully
 *                                       // transferred. If an error occurred
 *                                       // this will be less than the sum of
 *                                       // the sum of the buffer sizes.
 * ); @endcode
 *
 * @par Example:
 * To write a single data buffer use the @ref buffers function as follows:
 * @code asio::async_write_n(s, asio::buffers(data, size), handler); @endcode
 * See the @ref buffers documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename Async_Write_Stream, typename Const_Buffers, typename Handler>
inline void async_write_n(Async_Write_Stream& s, const Const_Buffers& buffers,
    Handler handler)
{
  async_write(s, buffers,
      detail::write_n_handler<Async_Write_Stream, Const_Buffers, Handler>(
        s, buffers, handler));
}

/*@}*/
/**
 * @defgroup write_at_least_n asio::write_at_least_n
 */
/*@{*/

/// Write at least a certain number of bytes of data to a stream before
/// returning.
/**
 * This function is used to write at least a certain number of bytes of data to
 * a stream. The call will block until one of the following conditions is true:
 *
 * @li The total bytes transferred is greater than or equal to the specified
 * minimum size.
 *
 * @li All data in the supplied buffers has been written. That is, the total
 * bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This function is implemented in terms of one or more calls to @ref write.
 * 
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param buffers One or more buffers containing the data to be written.
 *
 * @param min_length The minimum size of data to be written, in bytes.
 *
 * @param total_bytes_transferred An optional output parameter that receives the
 * total number of bytes actually written.
 *
 * @returns The number of bytes written on the last write.
 *
 * @note Throws an exception on failure. The type of the exception depends
 * on the underlying stream's write operation.
 *
 * @par Example:
 * To write a single data buffer use the @ref buffers function as follows:
 * @code asio::write_at_least_n(s,
 *     asio::buffers(data, size), min_length); @endcode
 * See the @ref buffers documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename Sync_Write_Stream, typename Const_Buffers>
std::size_t write_at_least_n(Sync_Write_Stream& s, const Const_Buffers& buffers,
    std::size_t min_length, std::size_t* total_bytes_transferred = 0)
{
  consuming_buffers<Const_Buffers> tmp(buffers);
  std::size_t bytes_transferred = 0;
  std::size_t total_transferred = 0;
  while (tmp.begin() != tmp.end() && total_transferred < min_length)
  {
    bytes_transferred = write(s, tmp);
    if (bytes_transferred == 0)
    {
      if (total_bytes_transferred)
        *total_bytes_transferred = total_transferred;
      return bytes_transferred;
    }
    tmp.consume(bytes_transferred);
    total_transferred += bytes_transferred;
  }
  if (total_bytes_transferred)
    *total_bytes_transferred = total_transferred;
  return bytes_transferred;
}

/// Write at least a certain number of bytes of data to a stream before
/// returning.
/**
 * This function is used to write at least a certain number of bytes of data to
 * a stream. The call will block until one of the following conditions is true:
 *
 * @li The total bytes transferred is greater than or equal to the specified
 * minimum size.
 *
 * @li All data in the supplied buffers has been written. That is, the total
 * bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This function is implemented in terms of one or more calls to @ref write.
 * 
 * @param s The stream to which the data is to be written. The type must support
 * the Sync_Write_Stream concept.
 *
 * @param buffers One or more buffers containing the data to be written.
 *
 * @param min_length The minimum size of data to be written, in bytes.
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
 * @returns The number of bytes written on the last write.
 */
template <typename Sync_Write_Stream, typename Const_Buffers,
    typename Error_Handler>
std::size_t write_at_least_n(Sync_Write_Stream& s, const Const_Buffers& buffers,
    std::size_t min_length, std::size_t* total_bytes_transferred,
    Error_Handler error_handler)
{
  consuming_buffers<Const_Buffers> tmp(buffers);
  std::size_t bytes_transferred = 0;
  std::size_t total_transferred = 0;
  while (tmp.begin() != tmp.end() && total_transferred < min_length)
  {
    bytes_transferred = write(s, tmp, error_handler);
    if (bytes_transferred == 0)
    {
      if (total_bytes_transferred)
        *total_bytes_transferred = total_transferred;
      return bytes_transferred;
    }
    tmp.consume(bytes_transferred);
    total_transferred += bytes_transferred;
  }
  if (total_bytes_transferred)
    *total_bytes_transferred = total_transferred;
  return bytes_transferred;
}

/*@}*/

namespace detail
{
  template <typename Async_Write_Stream, typename Const_Buffers,
      typename Handler>
  class write_at_least_n_handler
  {
  public:
    write_at_least_n_handler(Async_Write_Stream& stream,
        const Const_Buffers& buffers, std::size_t min_length, Handler handler)
      : stream_(stream),
        buffers_(buffers),
        min_length_(min_length),
        total_transferred_(0),
        handler_(handler)
    {
    }

    template <typename Error>
    void operator()(const Error& e, std::size_t bytes_transferred)
    {
      total_transferred_ += bytes_transferred;
      buffers_.consume(bytes_transferred);
      if (e || bytes_transferred == 0 || buffers_.begin() == buffers_.end()
          || total_transferred_ >= min_length_)
      {
        stream_.demuxer().dispatch(detail::bind_handler(handler_, e,
              bytes_transferred, total_transferred_));
      }
      else
      {
        asio::async_write(stream_, buffers_, *this);
      }
    }

  private:
    Async_Write_Stream& stream_;
    consuming_buffers<Const_Buffers> buffers_;
    std::size_t min_length_;
    std::size_t total_transferred_;
    Handler handler_;
  };
} // namespace detail

/**
 * @defgroup async_write_at_least_n asio::async_write_at_least_n
 */
/*@{*/

/// Start an asynchronous write of at least a certain number of bytes of data to
/// a stream.
/**
 * This function is used to asynchronously write at least a certain number of
 * bytes of data to a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li The total bytes transferred is greater than or equal to the specified
 * minimum size.
 *
 * @li All of the data in the supplied buffers has been written. That is, the
 * total bytes transferred is equal to the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This function is implemented in terms of one or more calls to @ref
 * async_write.
 * 
 * @param s The stream to which the data is to be written. The type must support
 * the Async_Write_Stream concept.
 *
 * @param buffers One or more buffers containing the data to be written.
 * Although the buffers object may be copied as necessary, ownership of the
 * underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param min_length The minimum size of data to be written, in bytes.
 *
 * @param handler The handler to be called when the write operation completes.
 * Copies will be made of the handler as required. The equivalent function
 * signature of the handler must be:
 * @code template <typename Error>
 * void handler(
 *   const Error& error,                 // Result of operation (the actual type
 *                                       // is dependent on the underlying
 *                                       // stream's write operation).
 *
 *   std::size_t last_bytes_transferred, // Number of bytes transferred on last
 *                                       // write operation.
 *
 *   std::size_t total_bytes_transferred // Total number of bytes successfully
 *                                       // transferred. If an error occurred
 *                                       // this will be less than the minimum
 *                                       // size.
 * ); @endcode
 *
 * @par Example:
 * To write a single data buffer use the @ref buffers function as follows:
 * @code asio::async_write_at_least_n(s,
 *     asio::buffers(data, size), min_length, handler); @endcode
 * See the @ref buffers documentation for information on writing multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename Async_Write_Stream, typename Const_Buffers, typename Handler>
inline void async_write_at_least_n(Async_Write_Stream& s,
    const Const_Buffers& buffers, std::size_t min_length, Handler handler)
{
  async_write(s, buffers,
      detail::write_at_least_n_handler<Async_Write_Stream, Const_Buffers,
          Handler>(s, buffers, min_length, handler));
}

/*@}*/

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_WRITE_HPP
