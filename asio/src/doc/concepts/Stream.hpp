//
// Stream.hpp
// ~~~~~~~~~~
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
