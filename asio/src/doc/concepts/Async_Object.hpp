//
// Async_Object.hpp
// ~~~~~~~~~~~~~~~~
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

/// Asynchronous object concept.
/**
 * @par Implemented By:
 * asio::basic_dgram_socket @n
 * asio::basic_locking_dispatcher @n
 * asio::basic_socket_acceptor @n
 * asio::basic_socket_connector @n
 * asio::basic_stream_socket @n
 * asio::basic_timer @n
 * asio::buffered_recv_stream @n
 * asio::buffered_send_stream @n
 * asio::buffered_stream
 */
class Async_Object
{
public:
  /// The demuxer type for this asynchronous object.
  typedef implementation_defined demuxer_type;

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the object
   * uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the demuxer object that the object will use to
   * dispatch handlers. Ownership is not transferred to the caller.
   */
  demuxer_type& demuxer();
};
