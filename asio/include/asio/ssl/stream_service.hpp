//
// stream_service.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2005 Voipster / Indrek dot Juhani at voipster dot com
// Copyright (c) 2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_STREAM_SERVICE_HPP
#define ASIO_SSL_STREAM_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <memory>
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/ssl/basic_context.hpp"
#include "asio/ssl/stream_base.hpp"
#include "asio/ssl/detail/openssl_stream_service.hpp"

namespace asio {
namespace ssl {

/// Default service implementation for an SSL stream.
template <typename Allocator = std::allocator<void> >
class stream_service
  : private boost::noncopyable
{
public:
  /// The demuxer type.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

private:
  // The type of the platform-specific implementation.
  typedef detail::openssl_stream_service<Allocator> service_impl_type;

public:
  /// The type of a stream implementation.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined impl_type;
#else
  typedef typename service_impl_type::impl_type impl_type;
#endif

  /// Construct a new stream service for the specified demuxer.
  explicit stream_service(demuxer_type& demuxer)
    : service_impl_(demuxer.get_service(service_factory<service_impl_type>()))
  {
  }

  /// Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return service_impl_.demuxer();
  }

  /// Return a null stream implementation.
  impl_type null() const
  {
    return service_impl_.null();
  }

  /// Create a new stream implementation.
  template <typename Stream, typename Context_Service>
  void create(impl_type& impl, Stream& next_layer,
      basic_context<Context_Service>& context)
  {
    service_impl_.create(impl, next_layer, context);
  }

  /// Destroy a stream implementation.
  template <typename Stream>
  void destroy(impl_type& impl, Stream& next_layer)
  {
    service_impl_.destroy(impl, next_layer);
  }

  /// Perform SSL handshaking.
  template <typename Stream, typename Error_Handler>
  void handshake(impl_type& impl, Stream& next_layer,
      stream_base::handshake_type type, Error_Handler error_handler)
  {
    service_impl_.handshake(impl, next_layer, type, error_handler);
  }

  /// Start an asynchronous SSL handshake.
  template <typename Stream, typename Handler>
  void async_handshake(impl_type& impl, Stream& next_layer,
      stream_base::handshake_type type, Handler handler)
  {
    service_impl_.async_handshake(impl, next_layer, type, handler);
  }

  /// Shut down SSL on the stream.
  template <typename Stream, typename Error_Handler>
  void shutdown(impl_type& impl, Stream& next_layer,
      Error_Handler error_handler)
  {
    service_impl_.shutdown(impl, next_layer, error_handler);
  }

  /// Asynchronously shut down SSL on the stream.
  template <typename Stream, typename Handler>
  void async_shutdown(impl_type& impl, Stream& next_layer, Handler handler)
  {
    service_impl_.async_shutdown(impl, next_layer, handler);
  }

  /// Write some data to the stream.
  template <typename Stream, typename Const_Buffers, typename Error_Handler>
  std::size_t write_some(impl_type& impl, Stream& next_layer,
      const Const_Buffers& buffers, Error_Handler error_handler)
  {
    return service_impl_.write_some(impl, next_layer, buffers, error_handler);
  }

  /// Start an asynchronous write.
  template <typename Stream, typename Const_Buffers, typename Handler>
  void async_write_some(impl_type& impl, Stream& next_layer,
      const Const_Buffers& buffers, Handler handler)
  {
    service_impl_.async_write_some(impl, next_layer, buffers, handler);
  }

  /// Read some data from the stream.
  template <typename Stream, typename Mutable_Buffers, typename Error_Handler>
  std::size_t read_some(impl_type& impl, Stream& next_layer,
      const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    return service_impl_.read_some(impl, next_layer, buffers, error_handler);
  }

  /// Start an asynchronous read.
  template <typename Stream, typename Mutable_Buffers, typename Handler>
  void async_read_some(impl_type& impl, Stream& next_layer,
      const Mutable_Buffers& buffers, Handler handler)
  {
    service_impl_.async_read_some(impl, next_layer, buffers, handler);
  }

  /// Peek at the incoming data on the stream.
  template <typename Stream, typename Mutable_Buffers, typename Error_Handler>
  std::size_t peek(impl_type& impl, Stream& next_layer,
      const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    return service_impl_.peek(impl, next_layer, buffers, error_handler);
  }

  /// Determine the amount of data that may be read without blocking.
  template <typename Stream, typename Error_Handler>
  std::size_t in_avail(impl_type& impl, Stream& next_layer,
      Error_Handler error_handler)
  {
    return service_impl_.in_avail(impl, next_layer, error_handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_STREAM_SERVICE_HPP
