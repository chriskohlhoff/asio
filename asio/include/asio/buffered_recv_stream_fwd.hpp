//
// buffered_recv_stream_fwd.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_BUFFERED_RECV_STREAM_FWD_HPP
#define ASIO_BUFFERED_RECV_STREAM_FWD_HPP

#include "asio/detail/push_options.hpp"

#include "asio/fixed_buffer.hpp"

namespace asio {

template <typename Next_Layer, typename Buffer = fixed_buffer<8192> >
class buffered_recv_stream;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_RECV_STREAM_FWD_HPP
