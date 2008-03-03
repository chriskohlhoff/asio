//
// read_until.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_READ_UNTIL_HPP
#define ASIO_READ_UNTIL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include <boost/regex.hpp>
#include <string>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_streambuf.hpp"
#include "asio/error.hpp"

namespace asio {

/**
 * @defgroup read_until asio::read_until
 */
/*@{*/

/// Read data into a streambuf until a delimiter is encountered.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains the specified delimiter. The call will block
 * until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains the
 * delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param delim The delimiter character.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the delimiter.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @par Example
 * To read data into a streambuf until a newline is encountered:
 * @code asio::streambuf b;
 * asio::read_until(s, b, '\n');
 * std::istream is(&b);
 * std::string line;
 * std::getline(is, line); @endcode
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, char delim);

/// Read data into a streambuf until a delimiter is encountered.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains the specified delimiter. The call will block
 * until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains the
 * delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param delim The delimiter character.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the delimiter. Returns 0 if an error occurred.
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, char delim,
    asio::error_code& ec);

/// Read data into a streambuf until a delimiter is encountered.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains the specified delimiter. The call will block
 * until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains the
 * delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param delim The delimiter string.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the delimiter.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @par Example
 * To read data into a streambuf until a newline is encountered:
 * @code asio::streambuf b;
 * asio::read_until(s, b, "\r\n");
 * std::istream is(&b);
 * std::string line;
 * std::getline(is, line); @endcode
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const std::string& delim);

/// Read data into a streambuf until a delimiter is encountered.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains the specified delimiter. The call will block
 * until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains the
 * delimiter, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param delim The delimiter string.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the delimiter. Returns 0 if an error occurred.
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const std::string& delim,
    asio::error_code& ec);

/// Read data into a streambuf until a regular expression is located.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains some data that matches a regular expression.
 * The call will block until one of the following conditions is true:
 *
 * @li A substring of the streambuf's get area matches the regular expression.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains data that
 * matches the regular expression, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param expr The regular expression.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the substring that matches the regular expression.
 *
 * @throws asio::system_error Thrown on failure.
 *
 * @par Example
 * To read data into a streambuf until a CR-LF sequence is encountered:
 * @code asio::streambuf b;
 * asio::read_until(s, b, boost::regex("\r\n"));
 * std::istream is(&b);
 * std::string line;
 * std::getline(is, line); @endcode
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const boost::regex& expr);

/// Read data into a streambuf until a regular expression is located.
/**
 * This function is used to read data into the specified streambuf until the
 * streambuf's get area contains some data that matches a regular expression.
 * The call will block until one of the following conditions is true:
 *
 * @li A substring of the streambuf's get area matches the regular expression.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * read_some function. If the streambuf's get area already contains data that
 * matches the regular expression, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the SyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read.
 *
 * @param expr The regular expression.
 *
 * @param ec Set to indicate what error occurred, if any.
 *
 * @returns The number of bytes in the streambuf's get area up to and including
 * the substring that matches the regular expression. Returns 0 if an error
 * occurred.
 */
template <typename SyncReadStream, typename Allocator>
std::size_t read_until(SyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const boost::regex& expr,
    asio::error_code& ec);

/*@}*/
/**
* @defgroup async_read_until asio::async_read_until
*/
/*@{*/

/// Start an asynchronous operation to read data into a streambuf until a
/// delimiter is encountered.
/**
 * This function is used to asynchronously read data into the specified
 * streambuf until the streambuf's get area contains the specified delimiter.
 * The function call always returns immediately. The asynchronous operation
 * will continue until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function. If the streambuf's get area already contains the
 * delimiter, the asynchronous operation completes immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read. Ownership of
 * the streambuf is retained by the caller, which must guarantee that it remains
 * valid until the handler is called.
 *
 * @param delim The delimiter character.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const asio::error_code& error,         // Result of operation.
 *
 *   std::size_t bytes_transferred          // The number of bytes in the
 *                                          // streambuf's get area up to
 *                                          // and including the delimiter.
 *                                          // 0 if an error occurred.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_service::post().
 *
 * @par Example
 * To asynchronously read data into a streambuf until a newline is encountered:
 * @code asio::streambuf b;
 * ...
 * void handler(const asio::error_code& e, std::size_t size)
 * {
 *   if (!e)
 *   {
 *     std::istream is(&b);
 *     std::string line;
 *     std::getline(is, line);
 *     ...
 *   }
 * }
 * ...
 * asio::async_read_until(s, b, '\n', handler); @endcode
 */
template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
void async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b,
    char delim, ReadHandler handler);

/// Start an asynchronous operation to read data into a streambuf until a
/// delimiter is encountered.
/**
 * This function is used to asynchronously read data into the specified
 * streambuf until the streambuf's get area contains the specified delimiter.
 * The function call always returns immediately. The asynchronous operation
 * will continue until one of the following conditions is true:
 *
 * @li The get area of the streambuf contains the specified delimiter.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function. If the streambuf's get area already contains the
 * delimiter, the asynchronous operation completes immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read. Ownership of
 * the streambuf is retained by the caller, which must guarantee that it remains
 * valid until the handler is called.
 *
 * @param delim The delimiter string.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const asio::error_code& error,         // Result of operation.
 *
 *   std::size_t bytes_transferred          // The number of bytes in the
 *                                          // streambuf's get area up to
 *                                          // and including the delimiter.
 *                                          // 0 if an error occurred.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_service::post().
 *
 * @par Example
 * To asynchronously read data into a streambuf until a newline is encountered:
 * @code asio::streambuf b;
 * ...
 * void handler(const asio::error_code& e, std::size_t size)
 * {
 *   if (!e)
 *   {
 *     std::istream is(&b);
 *     std::string line;
 *     std::getline(is, line);
 *     ...
 *   }
 * }
 * ...
 * asio::async_read_until(s, b, "\r\n", handler); @endcode
 */
template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
void async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const std::string& delim,
    ReadHandler handler);

/// Start an asynchronous operation to read data into a streambuf until a
/// regular expression is located.
/**
 * This function is used to asynchronously read data into the specified
 * streambuf until the streambuf's get area contains some data that matches a
 * regular expression. The function call always returns immediately. The
 * asynchronous operation will continue until one of the following conditions
 * is true:
 *
 * @li A substring of the streambuf's get area matches the regular expression.
 *
 * @li An error occurred.
 *
 * This operation is implemented in terms of zero or more calls to the stream's
 * async_read_some function. If the streambuf's get area already contains data
 * that matches the regular expression, the function returns immediately.
 *
 * @param s The stream from which the data is to be read. The type must support
 * the AsyncReadStream concept.
 *
 * @param b A streambuf object into which the data will be read. Ownership of
 * the streambuf is retained by the caller, which must guarantee that it remains
 * valid until the handler is called.
 *
 * @param expr The regular expression.
 *
 * @param handler The handler to be called when the read operation completes.
 * Copies will be made of the handler as required. The function signature of the
 * handler must be:
 * @code void handler(
 *   const asio::error_code& error,         // Result of operation.
 *
 *   std::size_t bytes_transferred          // The number of bytes in the
 *                                          // streambuf's get area up to
 *                                          // and including the substring
 *                                          // that matches the regular.
 *                                          // expression. 0 if an error
 *                                          // occurred.
 * ); @endcode
 * Regardless of whether the asynchronous operation completes immediately or
 * not, the handler will not be invoked from within this function. Invocation of
 * the handler will be performed in a manner equivalent to using
 * asio::io_service::post().
 *
 * @par Example
 * To asynchronously read data into a streambuf until a CR-LF sequence is
 * encountered:
 * @code asio::streambuf b;
 * ...
 * void handler(const asio::error_code& e, std::size_t size)
 * {
 *   if (!e)
 *   {
 *     std::istream is(&b);
 *     std::string line;
 *     std::getline(is, line);
 *     ...
 *   }
 * }
 * ...
 * asio::async_read_until(s, b, boost::regex("\r\n"), handler); @endcode
 */
template <typename AsyncReadStream, typename Allocator, typename ReadHandler>
void async_read_until(AsyncReadStream& s,
    asio::basic_streambuf<Allocator>& b, const boost::regex& expr,
    ReadHandler handler);

/*@}*/

} // namespace asio

#include "asio/impl/read_until.ipp"

#include "asio/detail/pop_options.hpp"

#endif // ASIO_READ_UNTIL_HPP
