//
// Async_Send_Stream.hpp
// ~~~~~~~~~~~~~~~~~~~~~
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

/// Asynchronous send stream concept.
/**
 * @par Implemented By:
 * asio::basic_stream_socket @n
 * asio::buffered_recv_stream @n
 * asio::buffered_send_stream @n
 * asio::buffered_stream
 */
class Async_Send_Stream
  : public Async_Object
{
public:
  /// Start an asynchronous send.
  /**
   * This function is used to asynchronously send data on the stream. The
   * function call always returns immediately.
   *
   * @param data The data to be sent on the stream. Ownership of the data is
   * retained by the caller, which must guarantee that it is valid until the
   * handler is called.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param handler The handler to be called when the send operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const implementation_defined& error, // Result of operation
   *   size_t bytes_sent                    // Number of bytes sent
   * ); @endcode
   */
  template <typename Handler>
  void async_send(const void* data, size_t length, Handler handler);
};
