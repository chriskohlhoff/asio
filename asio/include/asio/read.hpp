//
// read.hpp
// ~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_READ_HPP
#define ASIO_READ_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_streambuf.hpp"

namespace asio {

/**
 * @defgroup read asio::read
 */
/*@{*/

/// Attempt to read a certain amount of data from a stream before returning.
/**
 * This function is used to read a certain number of bytes of data from a
 * stream. The call will block until one of the following conditions is true:
 *
 * @li The supplied buffers are full. That is, the bytes transferred is equal to
 * the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of one or more calls to the stream's
 * read_some function.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Sync_Read_Stream concept.
 *
 * @param buffers One or more buffers into which the data will be read. The sum
 * of the buffer sizes indicates the maximum number of bytes to read from the
 * stream.
 *
 * @returns The number of bytes transferred.
 *
 * @throws Sync_Read_Stream::error_type Thrown on failure.
 *
 * @par Example:
 * To read into a single data buffer use the @ref buffer function as follows:
 * @code asio::read(s, asio::buffer(data, size)); @endcode
 * See the @ref buffer documentation for information on reading into multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 *
 * @note This overload is equivalent to calling:
 * @code asio::read(
 *     s, buffers,
 *     asio::transfer_all(),
 *     asio::throw_error()); @endcode
 */
template <typename Sync_Read_Stream, typename Mutable_Buffers>
std::size_t read(Sync_Read_Stream& s, const Mutable_Buffers& buffers);

/// Attempt to read a certain amount of data from a stream before returning.
/**
 * This function is used to read a certain number of bytes of data from a
 * stream. The call will block until one of the following conditions is true:
 *
 * @li The supplied buffers are full. That is, the bytes transferred is equal to
 * the sum of the buffer sizes.
 *
 * @li The completion_condition function object returns true.
 *
 * This operation is implemented in terms of one or more calls to the stream's
 * read_some function.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Sync_Read_Stream concept.
 *
 * @param buffers One or more buffers into which the data will be read. The sum
 * of the buffer sizes indicates the maximum number of bytes to read from the
 * stream.
 *
 * @param completion_condition The function object to be called to determine
 * whether the read operation is complete. The signature of the function object
 * must be:
 * @code bool completion_condition(
 *   const Sync_Read_Stream::error_type& error, // Result of latest read_some
 *                                              // operation.
 *
 *   std::size_t bytes_transferred              // Number of bytes transferred
 *                                              // so far.
 * ); @endcode
 * A return value of true indicates that the read operation is complete. False
 * indicates that further calls to the stream's read_some function are required.
 *
 * @returns The number of bytes transferred.
 *
 * @throws Sync_Read_Stream::error_type Thrown on failure.
 *
 * @par Example:
 * To read into a single data buffer use the @ref buffer function as follows:
 * @code asio::read(s, asio::buffer(data, size),
 *     asio::transfer_at_least(32)); @endcode
 * See the @ref buffer documentation for information on reading into multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 *
 * @note This overload is equivalent to calling:
 * @code asio::read(
 *     s, buffers, completion_condition,
 *     asio::throw_error()); @endcode
 */
template <typename Sync_Read_Stream, typename Mutable_Buffers,
  typename Completion_Condition>
std::size_t read(Sync_Read_Stream& s, const Mutable_Buffers& buffers,
    Completion_Condition completion_condition);

/// Attempt to read a certain amount of data from a stream before returning.
/**
 * This function is used to read a certain number of bytes of data from a
 * stream. The call will block until one of the following conditions is true:
 *
 * @li The supplied buffers are full. That is, the bytes transferred is equal to
 * the sum of the buffer sizes.
 *
 * @li The completion_condition function object returns true.
 *
 * This operation is implemented in terms of one or more calls to the stream's
 * read_some function.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Sync_Read_Stream concept.
 *
 * @param buffers One or more buffers into which the data will be read. The sum
 * of the buffer sizes indicates the maximum number of bytes to read from the
 * stream.
 *
 * @param completion_condition The function object to be called to determine
 * whether the read operation is complete. The signature of the function object
 * must be:
 * @code bool completion_condition(
 *   const Sync_Read_Stream::error_type& error, // Result of latest read_some
 *                                              // operation.
 *
 *   std::size_t bytes_transferred              // Number of bytes transferred
 *                                              // so far.
 * ); @endcode
 * A return value of true indicates that the read operation is complete. False
 * indicates that further calls to the stream's read_some function are required.
 *
 * @param error_handler A handler to be called when the operation completes,
 * to indicate whether or not an error has occurred. Copies will be made of
 * the handler as required. The function signature of the handler must be:
 * @code void error_handler(
 *   const Sync_Read_Stream::error_type& error // Result of operation.
 * ); @endcode
 * The error handler is only called if the completion_condition indicates that
 * the operation is complete.
 *
 * @returns The number of bytes read. If an error occurs, and the error handler
 * does not throw an exception, returns the total number of bytes successfully
 * transferred prior to the error.
 */
template <typename Sync_Read_Stream, typename Mutable_Buffers,
    typename Completion_Condition, typename Error_Handler>
std::size_t read(Sync_Read_Stream& s, const Mutable_Buffers& buffers,
    Completion_Condition completion_condition, Error_Handler error_handler);

/// Attempt to read a certain amount of data from a stream before returning.
/**
 * This function is used to read a certain number of bytes of data from a
 * stream. The call will block until one of the following conditions is true:
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of one or more calls to the stream's
 * read_some function.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Sync_Read_Stream concept.
 *
 * @param b The basic_streambuf object into which the data will be read.
 *
 * @returns The number of bytes transferred.
 *
 * @throws Sync_Read_Stream::error_type Thrown on failure.
 *
 * @note This overload is equivalent to calling:
 * @code asio::read(
 *     s, b,
 *     asio::transfer_all(),
 *     asio::throw_error()); @endcode
 */
template <typename Sync_Read_Stream, typename Allocator>
std::size_t read(Sync_Read_Stream& s, basic_streambuf<Allocator>& b);

/// Attempt to read a certain amount of data from a stream before returning.
/**
 * This function is used to read a certain number of bytes of data from a
 * stream. The call will block until one of the following conditions is true:
 *
 * @li The completion_condition function object returns true.
 *
 * This operation is implemented in terms of one or more calls to the stream's
 * read_some function.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Sync_Read_Stream concept.
 *
 * @param b The basic_streambuf object into which the data will be read.
 *
 * @param completion_condition The function object to be called to determine
 * whether the read operation is complete. The signature of the function object
 * must be:
 * @code bool completion_condition(
 *   const Sync_Read_Stream::error_type& error, // Result of latest read_some
 *                                              // operation.
 *
 *   std::size_t bytes_transferred              // Number of bytes transferred
 *                                              // so far.
 * ); @endcode
 * A return value of true indicates that the read operation is complete. False
 * indicates that further calls to the stream's read_some function are required.
 *
 * @returns The number of bytes transferred.
 *
 * @throws Sync_Read_Stream::error_type Thrown on failure.
 *
 * @note This overload is equivalent to calling:
 * @code asio::read(
 *     s, b, completion_condition,
 *     asio::throw_error()); @endcode
 */
template <typename Sync_Read_Stream, typename Allocator,
    typename Completion_Condition>
std::size_t read(Sync_Read_Stream& s, basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition);

/// Attempt to read a certain amount of data from a stream before returning.
/**
 * This function is used to read a certain number of bytes of data from a
 * stream. The call will block until one of the following conditions is true:
 *
 * @li The completion_condition function object returns true.
 *
 * This operation is implemented in terms of one or more calls to the stream's
 * read_some function.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Sync_Read_Stream concept.
 *
 * @param b The basic_streambuf object into which the data will be read.
 *
 * @param completion_condition The function object to be called to determine
 * whether the read operation is complete. The signature of the function object
 * must be:
 * @code bool completion_condition(
 *   const Sync_Read_Stream::error_type& error, // Result of latest read_some
 *                                              // operation.
 *
 *   std::size_t bytes_transferred              // Number of bytes transferred
 *                                              // so far.
 * ); @endcode
 * A return value of true indicates that the read operation is complete. False
 * indicates that further calls to the stream's read_some function are required.
 *
 * @param error_handler A handler to be called when the operation completes,
 * to indicate whether or not an error has occurred. Copies will be made of
 * the handler as required. The function signature of the handler must be:
 * @code void error_handler(
 *   const Sync_Read_Stream::error_type& error // Result of operation.
 * ); @endcode
 * The error handler is only called if the completion_condition indicates that
 * the operation is complete.
 *
 * @returns The number of bytes read. If an error occurs, and the error handler
 * does not throw an exception, returns the total number of bytes successfully
 * transferred prior to the error.
 */
template <typename Sync_Read_Stream, typename Allocator,
    typename Completion_Condition, typename Error_Handler>
std::size_t read(Sync_Read_Stream& s, basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition, Error_Handler error_handler);

/*@}*/
/**
 * @defgroup async_read asio::async_read
 */
/*@{*/

/// Start an asynchronous operation to read a certain amount of data from a
/// stream.
/**
 * This function is used to asynchronously read a certain number of bytes of
 * data from a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions is
 * true:
 *
 * @li The supplied buffers are full. That is, the bytes transferred is equal to
 * the sum of the buffer sizes.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of one or more calls to the stream's
 * async_read_some function.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Async_Read_Stream concept.
 *
 * @param buffers One or more buffers into which the data will be read. The sum
 * of the buffer sizes indicates the maximum number of bytes to read from the
 * stream. Although the buffers object may be copied as necessary, ownership of
 * the underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const Async_Read_Stream::error_type& error, // Result of operation.
 *
 *   std::size_t bytes_transferred               // Number of bytes copied into
 *                                               // the buffers. If an error
 *                                               // occurred, this will be the
 *                                               // number of bytes successfully
 *                                               // transferred prior to the
 *                                               // error.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_service::post().
 *
 * @par Example:
 * To read into a single data buffer use the @ref buffer function as follows:
 * @code
 * asio::async_read(s, asio::buffer(data, size), handler);
 * @endcode
 * See the @ref buffer documentation for information on reading into multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 *
 * @note This overload is equivalent to calling:
 * @code asio::async_read(
 *     s, buffers,
 *     asio::transfer_all(),
 *     handler); @endcode
 */
template <typename Async_Read_Stream, typename Mutable_Buffers,
    typename Handler>
void async_read(Async_Read_Stream& s, const Mutable_Buffers& buffers,
    Handler handler);

/// Start an asynchronous operation to read a certain amount of data from a
/// stream.
/**
 * This function is used to asynchronously read a certain number of bytes of
 * data from a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions is
 * true:
 *
 * @li The supplied buffers are full. That is, the bytes transferred is equal to
 * the sum of the buffer sizes.
 *
 * @li The completion_condition function object returns true.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Async_Read_Stream concept.
 *
 * @param buffers One or more buffers into which the data will be read. The sum
 * of the buffer sizes indicates the maximum number of bytes to read from the
 * stream. Although the buffers object may be copied as necessary, ownership of
 * the underlying memory blocks is retained by the caller, which must guarantee
 * that they remain valid until the handler is called.
 *
 * @param completion_condition The function object to be called to determine
 * whether the read operation is complete. The signature of the function object
 * must be:
 * @code bool completion_condition(
 *   const Async_Read_Stream::error_type& error, // Result of latest read_some
 *                                               // operation.
 *
 *   std::size_t bytes_transferred               // Number of bytes transferred
 *                                               // so far.
 * ); @endcode
 * A return value of true indicates that the read operation is complete. False
 * indicates that further calls to the stream's async_read_some function are
 * required.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const Async_Read_Stream::error_type& error, // Result of operation.
 *
 *   std::size_t bytes_transferred               // Number of bytes copied into
 *                                               // the buffers. If an error
 *                                               // occurred, this will be the
 *                                               // number of bytes successfully
 *                                               // transferred prior to the
 *                                               // error.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_service::post().
 *
 * @par Example:
 * To read into a single data buffer use the @ref buffer function as follows:
 * @code asio::async_read(s,
 *     asio::buffer(data, size),
 *     asio::transfer_at_least(32),
 *     handler); @endcode
 * See the @ref buffer documentation for information on reading into multiple
 * buffers in one go, and how to use it with arrays, boost::array or
 * std::vector.
 */
template <typename Async_Read_Stream, typename Mutable_Buffers,
    typename Completion_Condition, typename Handler>
void async_read(Async_Read_Stream& s, const Mutable_Buffers& buffers,
    Completion_Condition completion_condition, Handler handler);

/// Start an asynchronous operation to read a certain amount of data from a
/// stream.
/**
 * This function is used to asynchronously read a certain number of bytes of
 * data from a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions is
 * true:
 *
 * @li An error occurred.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Async_Read_Stream concept.
 *
 * This operation is implemented in terms of one or more calls to the stream's
 * async_read_some function.
 *
 * @param b A basic_streambuf object into which the data will be read. Ownership
 * of the streambuf is retained by the caller, which must guarantee that it
 * remains valid until the handler is called.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const Async_Read_Stream::error_type& error, // Result of operation.
 *
 *   std::size_t bytes_transferred               // Number of bytes copied into
 *                                               // the buffers. If an error
 *                                               // occurred, this will be the
 *                                               // number of bytes successfully
 *                                               // transferred prior to the
 *                                               // error.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_service::post().
 *
 * @note This overload is equivalent to calling:
 * @code asio::async_read(
 *     s, b,
 *     asio::transfer_all(),
 *     handler); @endcode
 */
template <typename Async_Read_Stream, typename Allocator, typename Handler>
void async_read(Async_Read_Stream& s, basic_streambuf<Allocator>& b,
    Handler handler);

/// Start an asynchronous operation to read a certain amount of data from a
/// stream.
/**
 * This function is used to asynchronously read a certain number of bytes of
 * data from a stream. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions is
 * true:
 *
 * @li The completion_condition function object returns true.
 *
 * This operation is implemented in terms of one or more calls to the stream's
 * async_read_some function.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the Async_Read_Stream concept.
 *
 * @param b A basic_streambuf object into which the data will be read. Ownership
 * of the streambuf is retained by the caller, which must guarantee that it
 * remains valid until the handler is called.
 *
 * @param completion_condition The function object to be called to determine
 * whether the read operation is complete. The signature of the function object
 * must be:
 * @code bool completion_condition(
 *   const Async_Read_Stream::error_type& error, // Result of latest read_some
 *                                               // operation.
 *
 *   std::size_t bytes_transferred               // Number of bytes transferred
 *                                               // so far.
 * ); @endcode
 * A return value of true indicates that the read operation is complete. False
 * indicates that further calls to the stream's async_read_some function are
 * required.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const Async_Read_Stream::error_type& error, // Result of operation.
 *
 *   std::size_t bytes_transferred               // Number of bytes copied into
 *                                               // the buffers. If an error
 *                                               // occurred, this will be the
 *                                               // number of bytes successfully
 *                                               // transferred prior to the
 *                                               // error.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_service::post().
 */
template <typename Async_Read_Stream, typename Allocator,
    typename Completion_Condition, typename Handler>
void async_read(Async_Read_Stream& s, basic_streambuf<Allocator>& b,
    Completion_Condition completion_condition, Handler handler);

/*@}*/

} // namespace asio

#include "asio/impl/read.ipp"

#include "asio/detail/pop_options.hpp"

#endif // ASIO_READ_HPP
