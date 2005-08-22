//
// Sync_Write_Stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Synchronous write stream concept.
/**
 * @par Implemented By:
 * asio::basic_stream_socket @n
 * asio::buffered_read_stream @n
 * asio::buffered_write_stream @n
 * asio::buffered_stream
 */
class Sync_Write_Stream
{
public:
  /// Write the given data on the stream.
  /**
   * This function is used to write data on the stream. The function call will
   * block until some or all of the data has been written successfully, or until
   * an error occurs.
   *
   * @param data The data to be written.
   *
   * @param length The size of the data to be written, in bytes.
   *
   * @returns The number of bytes written or 0 if the stream was closed cleanly.
   *
   * @throws implementation_specified Thrown on failure.
   */
  size_t write(const void* data, size_t length);

  /// Write the given data on the stream.
  /**
   * This function is used to write data on the stream. The function call will
   * block until some or all of the data has been written successfully, or until
   * an error occurs.
   *
   * @param data The data to be written.
   *
   * @param length The size of the data to be written, in bytes.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const implementation_defined& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes written or 0 if the stream was closed cleanly.
   */
  template <typename Error_Handler>
  size_t write(const void* data, size_t length, Error_Handler error_handler);
};
