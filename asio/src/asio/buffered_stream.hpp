//
// buffered_stream.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_BUFFERED_STREAM_HPP
#define ASIO_BUFFERED_STREAM_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffered_recv_stream.hpp"
#include "asio/buffered_send_stream.hpp"

namespace asio {

/// The buffered_stream class template can be used to add buffering to both the
/// send- and recv- related operations of a stream.
template <typename Stream>
class buffered_stream
    : private buffered_recv_stream<buffered_send_stream<Stream> >
{
public:
  typedef buffered_recv_stream<buffered_send_stream<Stream> > base_type;

  using base_type::close;
  using base_type::next_layer;
  using base_type::lowest_layer;
  using base_type::send;
  using base_type::async_send;
  using base_type::send_n;
  using base_type::async_send_n;
  using base_type::recv;
  using base_type::async_recv;
  using base_type::recv_n;
  using base_type::async_recv_n;

  /// Construct, passing the specified argument to initialise the next layer.
  template <typename Arg>
  explicit buffered_stream(Arg& a)
    : base_type(a)
  {
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_STREAM_HPP
