//
// Sync_Send_Stream.hpp
// ~~~~~~~~~~~~~~~~~~~~
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

/// Synchronous send stream concept.
/**
 * @par Implemented By:
 * asio::basic_stream_socket @n
 * asio::buffered_recv_stream @n
 * asio::buffered_send_stream @n
 * asio::buffered_stream
 */
class Sync_Send_Stream
{
public:
  /// Send the given data on the stream.
  /**
   * This function is used to send data on the stream. The function call will
   * block until some or all of the data has been sent successfully, or until
   * an error occurs.
   *
   * @param data The data to be sent.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @returns The number of bytes sent or 0 if the connection was closed
   * cleanly.
   *
   * @throws implementation_specified Thrown on failure.
   */
  size_t send(const void* data, size_t length);

  /// Send the given data on the stream.
  /**
   * This function is used to send data on the stream. The function call will
   * block until some or all of the data has been sent successfully, or until
   * an error occurs.
   *
   * @param data The data to be sent.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const implementation_defined& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes sent or 0 if the connection was closed
   * cleanly.
   */
  template <typename Error_Handler>
  size_t send(const void* data, size_t length, Error_Handler error_handler);
};
