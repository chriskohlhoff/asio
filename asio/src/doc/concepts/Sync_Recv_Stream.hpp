//
// Sync_Recv_Stream.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Synchronous receive stream concept.
/**
 * @par Implemented By:
 * asio::basic_stream_socket @n
 * asio::buffered_recv_stream @n
 * asio::buffered_send_stream @n
 * asio::buffered_stream
 */
class Sync_Recv_Stream
{
public:
  /// Receive some data from the stream.
  /**
   * This function is used to receive data from the stream. The function call
   * will block until data has received successfully or an error occurs.
   *
   * @param data The buffer into which the received data will be written.
   *
   * @param max_length The maximum size of the data to be received, in bytes.
   *
   * @returns The number of bytes received or 0 if the connection was closed
   * cleanly.
   *
   * @throws implementation_defined Thrown on failure.
   */
  size_t recv(void* data, size_t max_length);

  /// Receive some data from the stream.
  /**
   * This function is used to receive data from the stream. The function call
   * will block until data has received successfully or an error occurs.
   *
   * @param data The buffer into which the received data will be written.
   *
   * @param max_length The maximum size of the data to be received, in bytes.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const implementation_defined& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes received or 0 if the connection was closed
   * cleanly.
   */
  template <typename Error_Handler>
  size_t recv(void* data, size_t max_length, Error_Handler error_handler);
};
