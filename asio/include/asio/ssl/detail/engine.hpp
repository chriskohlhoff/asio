//
// ssl/detail/engine.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_ENGINE_HPP
#define ASIO_SSL_DETAIL_ENGINE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_ENABLE_OLD_SSL)
# include "asio/detail/static_mutex.hpp"
# include "asio/ssl/detail/buffer_space.hpp"
# include "asio/ssl/detail/openssl_types.hpp"
# include "asio/ssl/stream_base.hpp"
#endif // !defined(ASIO_ENABLE_OLD_SSL)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

#if !defined(ASIO_ENABLE_OLD_SSL)

class engine
{
public:
  // Construct a new engine for the specified context.
  ASIO_DECL explicit engine(SSL_CTX* context);

  // Destructor.
  ASIO_DECL ~engine();

  // Get the underlying implementation in the native type.
  ASIO_DECL SSL* native_handle();

  // Perform an SSL handshake using either SSL_connect (client-side) or
  // SSL_accept (server-side).
  ASIO_DECL int handshake(stream_base::handshake_type type,
      buffer_space& space, asio::error_code& ec);

  // Perform a graceful shutdown of the SSL session.
  ASIO_DECL int shutdown(buffer_space& space,
      asio::error_code& ec);

  // Write bytes to the SSL session.
  ASIO_DECL int write(const asio::const_buffer& data,
      buffer_space& space, asio::error_code& ec);

  // Read bytes from the SSL session.
  ASIO_DECL int read(const asio::mutable_buffer& data,
      buffer_space& space, asio::error_code& ec);

  // Map an error::eof code returned by the underlying transport according to
  // the type and state of the SSL session. Returns a const reference to the
  // error code object, suitable for passing to a completion handler.
  ASIO_DECL const asio::error_code& map_error_code(
      asio::error_code& ec) const;

private:
  // Disallow copying and assignment.
  engine(const engine&);
  engine& operator=(const engine&);

  // The SSL_accept function may not be thread safe. This mutex is used to
  // protect all calls to the SSL_accept function.
  ASIO_DECL static asio::detail::static_mutex& accept_mutex();

  // Perform one operation. Returns >= 0 on success or error, want_read if the
  // operation needs more input, or want_write if it needs to write some output
  // before the operation can complete.
  ASIO_DECL int perform(int (engine::* op)(void*, std::size_t),
      void* data, std::size_t length,
      buffer_space& space, asio::error_code& ec);

  // Adapt the SSL_accept function to the signature needed for perform().
  ASIO_DECL int do_accept(void*, std::size_t);

  // Adapt the SSL_connect function to the signature needed for perform().
  ASIO_DECL int do_connect(void*, std::size_t);

  // Adapt the SSL_shutdown function to the signature needed for perform().
  ASIO_DECL int do_shutdown(void*, std::size_t);

  // Adapt the SSL_read function to the signature needed for perform().
  ASIO_DECL int do_read(void* data, std::size_t length);

  // Adapt the SSL_write function to the signature needed for perform().
  ASIO_DECL int do_write(void* data, std::size_t length);

  SSL* ssl_;
  BIO* ext_bio_;
};

#endif // !defined(ASIO_ENABLE_OLD_SSL)

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ssl/detail/impl/engine.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_SSL_DETAIL_ENGINE_HPP
