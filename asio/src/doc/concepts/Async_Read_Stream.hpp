//
// Async_Read_Stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Asynchronous read stream concept.
/**
 * @par Implemented By:
 * asio::basic_stream_socket @n
 * asio::buffered_read_stream @n
 * asio::buffered_write_stream @n
 * asio::buffered_stream
 */
class Async_Read_Stream
  : public Async_Object
{
public:
  /// Start an asynchronous read.
  /**
   * This function is used to asynchronously read data from the stream. The
   * function call always returns immediately.
   *
   * @param data The buffer into which the data will be read. Ownership of the
   * buffer is retained by the caller, which must guarantee that it is valid
   * until the handler is called.
   *
   * @param max_length The maximum size of the data to be read, in bytes.
   *
   * @param handler The handler to be called when the read operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const implementation_defined& error, // Result of operation
   *   size_t bytes_transferred             // Number of bytes read
   * ); @endcode
   */
  template <typename Handler>
  void async_read(void* data, size_t max_length, Handler handler);
};
