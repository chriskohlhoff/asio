//
// buffered_write_stream_fwd.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BUFFERED_WRITE_STREAM_FWD_HPP
#define ASIO_BUFFERED_WRITE_STREAM_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/fixed_buffer.hpp"

namespace asio {

template <typename Stream, typename Buffer = fixed_buffer<8192> >
class buffered_write_stream;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BUFFERED_WRITE_STREAM_FWD_HPP
