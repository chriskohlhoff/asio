//
// is_send_buffered.hpp
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

#ifndef ASIO_IS_SEND_BUFFERED_HPP
#define ASIO_IS_SEND_BUFFERED_HPP

#include "asio/detail/push_options.hpp"

#include "asio/buffered_recv_stream_fwd.hpp"
#include "asio/buffered_stream_fwd.hpp"

namespace asio {

namespace detail {

template <typename Stream, typename Buffer>
char is_send_buffered_helper(buffered_stream<Stream, Buffer>* s);

template <typename Stream, typename Buffer>
char is_send_buffered_helper(buffered_send_stream<Stream, Buffer>* s);

struct is_send_buffered_big_type { char data[10]; };
is_send_buffered_big_type is_send_buffered_helper(...);

} // namespace detail

/// The is_send_buffered class is a traits class that may be used to determine
/// whether a stream type supports buffering of sent data.
template <typename Stream>
class is_send_buffered
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The value member is true only if the Stream type supports buffering of
  /// sent data.
  static const bool value;
#else
  enum { value = sizeof(detail::is_send_buffered_helper((Stream*)0)) == 1 };
#endif
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IS_SEND_BUFFERED_HPP
