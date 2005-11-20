//
// Sync_Write_Stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
 * asio::buffered_stream @n
 * asio::ssl::stream
 */
class Sync_Write_Stream
  : public Error_Source
{
public:
  /// Write the given data on the stream.
  /**
   * This function is used to write data on the stream. The function call will
   * block until one or more bytes of data has been written successfully, or
   * until an error occurs.
   *
   * @param buffers The data to be written.
   *
   * @returns The number of bytes written.
   *
   * @throws Sync_Write_Stream::error_type Thrown on failure.
   */
  template <typename Const_Buffers>
  std::size_t write_some(const Const_Buffers& buffers);

  /// Write the given data on the stream.
  /**
   * This function is used to write data on the stream. The function call will
   * block until one or more bytes of data has been written successfully, or
   * until an error occurs.
   *
   * @param buffers The data to be written.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const Sync_Write_Stream::error_type& error // Result of operation.
   * ); @endcode
   *
   * @returns The number of bytes written. Returns 0 if an error occurred and
   * the error handler did not throw an exception.
   */
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t write_some(const Const_Buffers& buffers,
      Error_Handler error_handler);
};
