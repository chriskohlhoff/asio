//
// Stream.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

/// Stream concept.
/**
 * @par Implemented By:
 * asio::basic_stream_socket @n
 * asio::buffered_read_stream @n
 * asio::buffered_write_stream @n
 * asio::buffered_stream @n
 * asio::ssl::stream
 */
class Stream
  : public Async_Read_Stream,
    public Async_Write_Stream,
    public Sync_Read_Stream,
    public Sync_Write_Stream
{
public:
  /// The type of the lowest layer in the stream.
  typedef implementation_defined lowest_layer_type;

  /// Get a reference to the lowest layer.
  lowest_layer_type& lowest_layer();
};
