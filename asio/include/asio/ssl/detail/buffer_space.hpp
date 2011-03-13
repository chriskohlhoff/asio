//
// ssl/detail/buffer_space.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_BUFFER_SPACE_HPP
#define ASIO_SSL_DETAIL_BUFFER_SPACE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_ENABLE_OLD_SSL)
# include <vector>
# include "asio/buffer.hpp"
#endif // !defined(ASIO_ENABLE_OLD_SSL)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

#if !defined(ASIO_ENABLE_OLD_SSL)

struct buffer_space
{
  // Returned by functions to indicate that the engine wants input. The input
  // buffer should be updated to point to the data.
  static const int want_input = -1;

  // Returned by functions to indicate that the engine wants to write output.
  // The output buffer points to the data to be written.
  static const int want_output = -2;

  // A buffer that may be used to prepare output intended for the transport.
  std::vector<unsigned char> output_buffer; 

  // The buffer pointing to the data to be written by the transport.
  asio::const_buffer output;

  // A buffer that may be used to read input intended for the engine.
  std::vector<unsigned char> input_buffer; 

  // The buffer pointing to the engine's unconsumed input.
  asio::const_buffer input;

  // Constructor sets up the buffers.
  buffer_space()
    : output_buffer(16384),
      input_buffer(16384)
  {
  }
};

#endif // !defined(ASIO_ENABLE_OLD_SSL)

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_BUFFER_SPACE_HPP
