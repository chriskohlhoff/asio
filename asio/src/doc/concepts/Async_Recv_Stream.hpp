//
// Async_Recv_Stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Asynchronous receive stream concept.
/**
 * @par Implemented By:
 * asio::basic_stream_socket @n
 * asio::buffered_recv_stream @n
 * asio::buffered_send_stream @n
 * asio::buffered_stream
 */
class Async_Recv_Stream
  : public Async_Object
{
public:
  /// Start an asynchronous receive.
  /**
   * This function is used to asynchronously receive data from the stream. The
   * function call always returns immediately.
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
   * @code void handler(
   *   const implementation_defined& error, // Result of operation
   *   size_t bytes_recvd                   // Number of bytes received
   * ); @endcode
   */
  template <typename Handler>
  void async_recv(void* data, size_t max_length, Handler handler);
};
