//
// Stream.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Stream concept.
/**
 * @par Implemented By:
 * asio::basic_stream_socket @n
 * asio::buffered_recv_stream @n
 * asio::buffered_send_stream @n
 * asio::buffered_stream
 */
class Stream
  : public Async_Recv_Stream,
    public Async_Send_Stream,
    public Sync_Recv_Stream,
    public Sync_Send_Stream
{
public:
  /// The type of the lowest layer in the stream.
  typedef implementation_defined lowest_layer_type;

  /// Get a reference to the lowest layer.
  lowest_layer_type& lowest_layer();
};
