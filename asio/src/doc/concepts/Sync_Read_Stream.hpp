//
// Sync_Read_Stream.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Synchronous read stream concept.
/**
 * @par Implemented By:
 * asio::basic_stream_socket @n
 * asio::buffered_read_stream @n
 * asio::buffered_write_stream @n
 * asio::buffered_stream @n
 * asio::ssl::stream
 */
class Sync_Read_Stream
  : public Error_Source
{
public:
  /// Read some data from the stream.
  /**
   * This function is used to read data from the stream. The function call will
   * block until one or more bytes of data has been read successfully, or until
   * an error occurs.
   *
   * @param buffers The buffers into which the data will be read.
   *
   * @returns The number of bytes read.
   *
   * @throws Sync_Read_Stream::error_type Thrown on failure.
   */
  template <typename Mutable_Buffers>
  std::size_t read_some(const Mutable_Buffers& buffers);

  /// Read some data from the stream.
  /**
   * This function is used to read data from the stream. The function call will
   * block until one or more bytes of data has been read successfully, or until
   * an error occurs.
   *
   * @param buffers The buffers into which the data will be read.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const Sync_Read_Stream::error_type& error // Result of operation.
   * ); @endcode
   *
   * @returns The number of bytes read. Returns 0 if an error occurred and the
   * error handler did not throw an exception.
   */
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t read_some(const Mutable_Buffers& buffers,
      Error_Handler error_handler);

  /// Peek at the incoming data on the stream.
  /**
   * This function is used to peek at the incoming data on the stream, without
   * removing it from the input queue. The function call will block until data
   * has been read successfully or an error occurs.
   *
   * @param buffers The buffers into which the data will be read.
   *
   * @returns The number of bytes read.
   *
   * @throws Sync_Read_Stream::error_type Thrown on failure.
   */
  template <typename Mutable_Buffers>
  std::size_t peek(const Mutable_Buffers& buffers);

  /// Peek at the incoming data on the stream.
  /**
   * This function is used to peek at the incoming data on the stream, without
   * removing it from the input queue. The function call will block until data
   * has been read successfully or an error occurs.
   *
   * @param buffers The buffers into which the data will be read.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const Sync_Read_Stream::error_type& error // Result of operation.
   * ); @endcode
   *
   * @returns The number of bytes read. Returns 0 if an error occurred and the
   * error handler did not throw an exception.
   */
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t peek(const Mutable_Buffers& buffers, Error_Handler error_handler);

  /// Determine the amount of data that may be read without blocking.
  /**
   * The function is used to determine the amount of data, in bytes, that may
   * be read from the stream without blocking.
   *
   * @returns The number of bytes of data that can be read without blocking.
   *
   * @throws Sync_Read_Stream::error_type Thrown on failure.
   */
  std::size_t in_avail();

  /// Determine the amount of data that may be read without blocking.
  /**
   * The function is used to determine the amount of data, in bytes, that may
   * be read from the stream without blocking.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const Sync_Read_Stream::error_type& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes of data that can be read without blocking.
   */
  template <typename Error_Handler>
  std::size_t in_avail(Error_Handler error_handler);
};
