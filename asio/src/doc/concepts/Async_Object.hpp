//
// Async_Object.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
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
