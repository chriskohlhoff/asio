//
// ssl/detail/stream_core.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2012 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_STREAM_CORE_HPP
#define ASIO_SSL_DETAIL_STREAM_CORE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_ENABLE_OLD_SSL)
# include "asio/deadline_timer.hpp"
# include "asio/ssl/detail/engine.hpp"
# include "asio/buffer.hpp"
#endif // !defined(ASIO_ENABLE_OLD_SSL)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

#if !defined(ASIO_ENABLE_OLD_SSL)

struct stream_core
{
  // According to the OpenSSL documentation, this is the buffer size that is is
  // sufficient to hold the largest possible TLS record.
  enum { max_tls_record_size = 17 * 1024 };

  stream_core(SSL_CTX* context, asio::io_service& io_service)
    : engine_(context),
      pending_read_(io_service),
      pending_write_(io_service),
      output_buffer_space_(max_tls_record_size),
      output_buffer_(asio::buffer(output_buffer_space_)),
      input_buffer_space_(max_tls_record_size),
      input_buffer_(asio::buffer(input_buffer_space_))
  {
    pending_read_.expires_at(boost::posix_time::neg_infin);
    pending_write_.expires_at(boost::posix_time::neg_infin);
  }

  ~stream_core()
  {
  }

  // The SSL engine.
  engine engine_;

  // Timer used for storing queued read operations.
  asio::deadline_timer pending_read_;

  // Timer used for storing queued write operations.
  asio::deadline_timer pending_write_;

  // Buffer space used to prepare output intended for the transport.
  std::vector<unsigned char> output_buffer_space_; 

  // A buffer that may be used to prepare output intended for the transport.
  const asio::mutable_buffers_1 output_buffer_; 

  // Buffer space used to read input intended for the engine.
  std::vector<unsigned char> input_buffer_space_; 

  // A buffer that may be used to read input intended for the engine.
  const asio::mutable_buffers_1 input_buffer_;

  // The list of buffers pointing to the engine's unconsumed input.
  std::list<asio::const_buffer> inputs_;
};

#endif // !defined(ASIO_ENABLE_OLD_SSL)

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_STREAM_CORE_HPP
