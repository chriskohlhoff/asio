//
// stream.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2005 Voipster / Indrek dot Juhani at voipster dot com
// Copyright (c) 2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_STREAM_HPP
#define ASIO_SSL_STREAM_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/type_traits.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/error_handler.hpp"
#include "asio/service_factory.hpp"
#include "asio/ssl/basic_context.hpp"
#include "asio/ssl/stream_base.hpp"
#include "asio/ssl/stream_service.hpp"

namespace asio {
namespace ssl {

/// Provides stream-oriented functionality using SSL.
/**
 * The stream class template provides asynchronous and blocking stream-oriented
 * functionality using SSL.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Example:
 * To use the SSL stream template with a stream_socket, you would write:
 * @code
 * asio::demuxer d;
 * asio::ssl::context context(d, asio::ssl::context::sslv23);
 * asio::ssl::stream<asio::stream_socket> sock(demuxer, context);
 * @endcode
 *
 * @par Concepts:
 * Async_Object, Async_Read_Stream, Async_Write_Stream, Error_Source, Stream,
 * Sync_Read_Stream, Sync_Write_Stream.
 */
template <typename Stream, typename Service = stream_service<> >
class stream
  : public stream_base,
    private boost::noncopyable
{
public:
  /// The type of the next layer.
  typedef typename boost::remove_reference<Stream>::type next_layer_type;

  /// The type of the lowest layer.
  typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

  /// The demuxer type for this asynchronous type.
  typedef typename next_layer_type::demuxer_type demuxer_type;

  /// The type used for reporting errors.
  typedef typename next_layer_type::error_type error_type;

  /// The type of the service that will be used to provide stream operations.
  typedef Service service_type;

  /// The native implementation type of the stream.
  typedef typename service_type::impl_type impl_type;

  /// Construct a stream.
  /**
   * This constructor creates a stream and initialises the underlying stream
   * object.
   *
   * @param arg The argument to be passed to initialise the underlying stream.
   *
   * @param context The SSL context to be used for the stream.
   */
  template <typename Arg, typename Context_Service>
  explicit stream(Arg& arg, basic_context<Context_Service>& context)
    : next_layer_(arg),
      service_(next_layer_.demuxer().get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_, next_layer_, context);
  }

  /// Destructor.
  ~stream()
  {
    service_.destroy(impl_, next_layer_);
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the stream uses
   * to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the demuxer object that stream will use to dispatch
   * handlers. Ownership is not transferred to the caller.
   */
  demuxer_type& demuxer()
  {
    return next_layer_.demuxer();
  }

  /// Get a reference to the next layer.
  /**
   * This function returns a reference to the next layer in a stack of stream
   * layers.
   *
   * @return A reference to the next layer in the stack of stream layers.
   * Ownership is not transferred to the caller.
   */
  next_layer_type& next_layer()
  {
    return next_layer_;
  }

  /// Get a reference to the lowest layer.
  /**
   * This function returns a reference to the lowest layer in a stack of
   * stream layers.
   *
   * @return A reference to the lowest layer in the stack of stream layers.
   * Ownership is not transferred to the caller.
   */
  lowest_layer_type& lowest_layer()
  {
    return next_layer_.lowest_layer();
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * context. This is intended to allow access to stream functionality that is
   * not otherwise provided.
   */
  impl_type impl()
  {
    return impl_;
  }

  /// Perform SSL handshaking.
  /**
   * This function is used to perform SSL handshaking on the stream. The
   * function call will block until handshaking is complete or an error occurs.
   *
   * @param type The type of handshaking to be performed, i.e. as a client or as
   * a server.
   *
   * @throws asio::error Thrown on failure.
   */
  void handshake(handshake_type type)
  {
    service_.handshake(impl_, next_layer_, type, throw_error());
  }

  /// Perform SSL handshaking.
  /**
   * This function is used to perform SSL handshaking on the stream. The
   * function call will block until handshaking is complete or an error occurs.
   *
   * @param type The type of handshaking to be performed, i.e. as a client or as
   * a server.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void handshake(handshake_type type, Error_Handler error_handler)
  {
    service_.handshake(impl_, next_layer_, type, error_handler);
  }

  /// Start an asynchronous SSL handshake.
  /**
   * This function is used to asynchronously perform an SSL handshake on the
   * stream. This function call always returns immediately.
   *
   * @param type The type of handshaking to be performed, i.e. as a client or as
   * a server.
   *
   * @param handler The handler to be called when the handshake operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation
   * ); @endcode
   */
  template <typename Handler>
  void async_handshake(handshake_type type, Handler handler)
  {
    service_.async_handshake(impl_, next_layer_, type, handler);
  }

  /// Shut down SSL on the stream.
  /**
   * This function is used to shut down SSL on the stream. The function call
   * will block until SSL has been shut down or an error occurs.
   *
   * @throws asio::error Thrown on failure.
   */
  void shutdown()
  {
    service_.shutdown(impl_, next_layer_, throw_error());
  }

  /// Shut down SSL on the stream.
  /**
   * This function is used to shut down SSL on the stream. The function call
   * will block until SSL has been shut down or an error occurs.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void shutdown(Error_Handler error_handler)
  {
    service_.shutdown(impl_, next_layer_, error_handler);
  }

  /// Asynchronously shut down SSL on the stream.
  /**
   * This function is used to asynchronously shut down SSL on the stream. This
   * function call always returns immediately.
   *
   * @param handler The handler to be called when the handshake operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation
   * ); @endcode
   */
  template <typename Handler>
  void async_shutdown(Handler handler)
  {
    service_.async_shutdown(impl_, next_layer_, handler);
  }

  /// Write some data to the stream.
  /**
   * This function is used to write data on the stream. The function call will
   * block until one or more bytes of data has been written successfully, or
   * until an error occurs.
   *
   * @param buffers The data to be written.
   *
   * @returns The number of bytes written.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note The write_some operation may not transmit all of the data to the
   * peer. Consider using the @ref write function if you need to ensure that all
   * data is written before the blocking operation completes.
   */
  template <typename Const_Buffers>
  std::size_t write_some(const Const_Buffers& buffers)
  {
    return service_.write_some(impl_, next_layer_, buffers, throw_error());
  }

  /// Write some data to the stream.
  /**
   * This function is used to write data on the stream. The function call will
   * block until one or more bytes of data has been written successfully, or
   * until an error occurs.
   *
   * @param buffers The data to be written to the stream.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation.
   * ); @endcode
   *
   * @returns The number of bytes written. Returns 0 if an error occurred and
   * the error handler did not throw an exception.
   *
   * @note The write_some operation may not transmit all of the data to the
   * peer. Consider using the @ref write function if you need to ensure that all
   * data is written before the blocking operation completes.
   */
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t write_some(const Const_Buffers& buffers,
      Error_Handler error_handler)
  {
    return service_.write_some(impl_, next_layer_, buffers, error_handler);
  }

  /// Start an asynchronous write.
  /**
   * This function is used to asynchronously write one or more bytes of data to
   * the stream. The function call always returns immediately.
   *
   * @param buffers The data to be written to the stream. Although the buffers
   * object may be copied as necessary, ownership of the underlying buffers is
   * retained by the caller, which must guarantee that they remain valid until
   * the handler is called.
   *
   * @param handler The handler to be called when the write operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes written.
   * ); @endcode
   *
   * @note The async_write_some operation may not transmit all of the data to
   * the peer. Consider using the @ref async_write function if you need to
   * ensure that all data is written before the blocking operation completes.
   */
  template <typename Const_Buffers, typename Handler>
  void async_write_some(const Const_Buffers& buffers, Handler handler)
  {
    service_.async_write_some(impl_, next_layer_, buffers, handler);
  }

  /// Read some data from the stream.
  /**
   * This function is used to read data from the stream. The function call will
   * block until one or more bytes of data has been read successfully, or until
   * an error occurs.
   *
   * @param buffers The buffers into which the data will be read.
   *
   * @returns The number of bytes read.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note The read_some operation may not read all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that the
   * requested amount of data is read before the blocking operation completes.
   */
  template <typename Mutable_Buffers>
  std::size_t read_some(const Mutable_Buffers& buffers)
  {
    return service_.read_some(impl_, next_layer_, buffers, throw_error());
  }

  /// Read some data from the stream.
  /**
   * This function is used to read data from the stream. The function call will
   * block until one or more bytes of data has been read successfully, or until
   * an error occurs.
   *
   * @param buffers The buffers into which the data will be read.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation.
   * ); @endcode
   *
   * @returns The number of bytes read. Returns 0 if an error occurred and the
   * error handler did not throw an exception.
   *
   * @note The read_some operation may not read all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that the
   * requested amount of data is read before the blocking operation completes.
   */
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t read_some(const Mutable_Buffers& buffers,
      Error_Handler error_handler)
  {
    return service_.read_some(impl_, next_layer_, buffers, error_handler);
  }

  /// Start an asynchronous read.
  /**
   * This function is used to asynchronously read one or more bytes of data from
   * the stream. The function call always returns immediately.
   *
   * @param buffers The buffers into which the data will be read. Although the
   * buffers object may be copied as necessary, ownership of the underlying
   * buffers is retained by the caller, which must guarantee that they remain
   * valid until the handler is called.
   *
   * @param handler The handler to be called when the read operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes read.
   * ); @endcode
   *
   * @note The async_read_some operation may not read all of the requested
   * number of bytes. Consider using the @ref async_read function if you need to
   * ensure that the requested amount of data is read before the asynchronous
   * operation completes.
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_read_some(const Mutable_Buffers& buffers, Handler handler)
  {
    service_.async_read_some(impl_, next_layer_, buffers, handler);
  }

  /// Peek at the incoming data on the stream.
  /**
   * This function is used to peek at the incoming data on the stream, without
   * removing it from the input queue. The function call will block until data
   * has been read successfully or an error occurs.
   *
   * @param buffers The buffers into which the data will be read.
   *
   * @returns The number of bytes read.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Mutable_Buffers>
  std::size_t peek(const Mutable_Buffers& buffers)
  {
    return service_.peek(impl_, next_layer_, buffers, throw_error());
  }

  /// Peek at the incoming data on the stream.
  /**
   * This function is used to peek at the incoming data on the stream, withoutxi
   * removing it from the input queue. The function call will block until data
   * has been read successfully or an error occurs.
   *
   * @param buffers The buffers into which the data will be read.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation.
   * ); @endcode
   *
   * @returns The number of bytes read. Returns 0 if an error occurred and the
   * error handler did not throw an exception.
   */
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t peek(const Mutable_Buffers& buffers, Error_Handler error_handler)
  {
    return service_.peek(impl_, next_layer_, buffers, error_handler);
  }

  /// Determine the amount of data that may be read without blocking.
  /**
   * This function is used to determine the amount of data, in bytes, that may
   * be read from the stream without blocking.
   *
   * @returns The number of bytes of data that can be read without blocking.
   *
   * @throws asio::error Thrown on failure.
   */
  std::size_t in_avail()
  {
    return service_.in_avail(impl_, next_layer_, throw_error());
  }

  /// Determine the amount of data that may be read without blocking.
  /**
   * This function is used to determine the amount of data, in bytes, that may
   * be read from the stream without blocking.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes of data that can be read without blocking.
   */
  template <typename Error_Handler>
  std::size_t in_avail(Error_Handler error_handler)
  {
    return service_.in_avail(impl_, next_layer_, error_handler);
  }

private:
  /// The next layer.
  Stream next_layer_;

  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_STREAM_HPP
