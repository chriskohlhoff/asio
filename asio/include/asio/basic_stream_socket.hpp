//
// basic_stream_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_STREAM_SOCKET_HPP
#define ASIO_BASIC_STREAM_SOCKET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/default_error_handler.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_base.hpp"

namespace asio {

/// Provides stream-oriented socket functionality.
/**
 * The basic_stream_socket class template provides asynchronous and blocking
 * stream-oriented socket functionality.
 *
 * Most applications will use the asio::stream_socket typedef.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object, Async_Read_Stream, Async_Write_Stream, Stream,
 * Sync_Read_Stream, Sync_Write_Stream.
 */
template <typename Service>
class basic_stream_socket
  : public socket_base,
    private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide socket operations.
  typedef Service service_type;

  /// The native implementation type of the stream socket.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// A basic_stream_socket is always the lowest layer.
  typedef basic_stream_socket<service_type> lowest_layer_type;

  /// Construct a basic_stream_socket without opening it.
  /**
   * This constructor creates a stream socket without connecting it to a remote
   * peer. The socket needs to be connected or accepted before data can be sent
   * or received on it.
   *
   * @param d The demuxer object that the stream socket will use to dispatch
   * handlers for any asynchronous operations performed on the socket.
   */
  explicit basic_stream_socket(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
  }

  /// Destructor.
  ~basic_stream_socket()
  {
    service_.close(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the stream
   * socket uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the demuxer object that stream socket will use to
   * dispatch handlers. Ownership is not transferred to the caller.
   */
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Close the socket.
  /**
   * This function is used to close the stream socket. Any asynchronous read
   * or write operations will be cancelled immediately.
   */
  void close()
  {
    service_.close(impl_);
  }

  /// Get a reference to the lowest layer.
  /**
   * This function returns a reference to the lowest layer in a stack of
   * stream layers. Since a basic_stream_socket cannot contain any further
   * stream layers, it simply returns a reference to itself.
   *
   * @return A reference to the lowest layer in the stack of stream layers.
   * Ownership is not transferred to the caller.
   */
  lowest_layer_type& lowest_layer()
  {
    return *this;
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * stream socket. This is intended to allow access to native socket
   * functionality that is not otherwise provided.
   */
  impl_type impl()
  {
    return impl_;
  }

  /// Set the underlying implementation in the native type.
  /**
   * This function is used by the acceptor and connector implementations to set
   * the underlying implementation associated with the stream socket.
   *
   * @param new_impl The new underlying socket implementation.
   */
  void set_impl(impl_type new_impl)
  {
    service_.open(impl_, new_impl);
  }

  /// Set an option on the socket.
  /**
   * This function is used to set an option on the socket.
   *
   * @param option The new option value to be set on the socket.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Option>
  void set_option(const Option& option)
  {
    service_.set_option(impl_, option, default_error_handler());
  }

  /// Set an option on the socket.
  /**
   * This function is used to set an option on the socket.
   *
   * @param option The new option value to be set on the socket.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Option, typename Error_Handler>
  void set_option(const Option& option, Error_Handler error_handler)
  {
    service_.set_option(impl_, option, error_handler);
  }

  /// Get an option from the socket.
  /**
   * This function is used to get the current value of an option on the socket.
   *
   * @param option The option value to be obtained from the socket.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Option>
  void get_option(Option& option) const
  {
    service_.get_option(impl_, option, default_error_handler());
  }

  /// Get an option from the socket.
  /**
   * This function is used to get the current value of an option on the socket.
   *
   * @param option The option value to be obtained from the socket.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Option, typename Error_Handler>
  void get_option(Option& option, Error_Handler error_handler) const
  {
    service_.get_option(impl_, option, error_handler);
  }

  /// Get the local endpoint of the socket.
  /**
   * This function is used to obtain the locally bound endpoint of the socket.
   *
   * @param endpoint An endpoint object that receives the local endpoint of the
   * socket.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Endpoint>
  void get_local_endpoint(Endpoint& endpoint) const
  {
    service_.get_local_endpoint(impl_, endpoint, default_error_handler());
  }

  /// Get the local endpoint of the socket.
  /**
   * This function is used to obtain the locally bound endpoint of the socket.
   *
   * @param endpoint An endpoint object that receives the local endpoint of the
   * socket.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    service_.get_local_endpoint(impl_, endpoint, error_handler);
  }

  /// Get the remote endpoint of the socket.
  /**
   * This function is used to obtain the remote endpoint of the socket.
   *
   * @param endpoint An endpoint object that receives the remote endpoint of
   * the socket.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Endpoint>
  void get_remote_endpoint(Endpoint& endpoint) const
  {
    service_.get_remote_endpoint(impl_, endpoint, default_error_handler());
  }

  /// Get the remote endpoint of the socket.
  /**
   * This function is used to obtain the remote endpoint of the socket.
   *
   * @param endpoint An endpoint object that receives the remote endpoint of
   * the socket.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Endpoint, typename Error_Handler>
  void get_remote_endpoint(Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    service_.get_remote_endpoint(impl_, endpoint, error_handler);
  }

  /// Disable sends or receives on the socket.
  /**
   * This function is used to disable send operations, receive operations, or
   * both.
   *
   * @param what Determines what types of operation will no longer be allowed.
   *
   * @throws asio::error Thrown on failure.
   */
  void shutdown(shutdown_type what)
  {
    service_.shutdown(impl_, what, default_error_handler());
  }

  /// Write some data to the socket.
  /**
   * This function is used to write data to the stream socket. The function call
   * will block until the data has been sent successfully or an error occurs.
   *
   * @param data The data to be written to the socket.
   *
   * @param length The size of the data to be written, in bytes.
   *
   * @returns The number of bytes written or 0 if the connection was closed
   * cleanly.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note The write operation may not transmit all of the data to the peer.
   * Consider using the asio::write_n() function if you need to ensure that all
   * data is written before the blocking operation completes.
   */
  size_t write(const void* data, size_t length)
  {
    return service_.send(impl_, data, length,
        message_flags(0), default_error_handler());
  }

  /// Write some data to the socket.
  /**
   * This function is used to write data to the stream socket. The function call
   * will block until the data has been sent successfully or an error occurs.
   *
   * @param data The data to be written to the socket.
   *
   * @param length The size of the data to be written, in bytes.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes sent or 0 if the connection was closed
   * cleanly.
   *
   * @note The write operation may not transmit all of the data to the peer.
   * Consider using the asio::write_n() function if you need to ensure that all
   * data is written before the blocking operation completes.
   */
  template <typename Error_Handler>
  size_t write(const void* data, size_t length, Error_Handler error_handler)
  {
    return service_.send(impl_, data, length, message_flags(0), error_handler);
  }

  /// Start an asynchronous write.
  /**
   * This function is used to asynchronously write data to the stream socket.
   * The function call always returns immediately.
   *
   * @param data The data to be written to the socket. Ownership of the data is
   * retained by the caller, which must guarantee that it is valid until the
   * handler is called.
   *
   * @param length The size of the data to be written, in bytes.
   *
   * @param handler The handler to be called when the write operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error, // Result of operation
   *   size_t bytes_transferred  // Number of bytes written
   * ); @endcode
   *
   * @note The write operation may not transmit all of the data to the peer.
   * Consider using the asio::async_write_n() function if you need to ensure
   * that all data is written before the asynchronous operation completes.
   */
  template <typename Handler>
  void async_write(const void* data, size_t length, Handler handler)
  {
    service_.async_send(impl_, data, length, message_flags(0), handler);
  }

  /// Read some data from the socket.
  /**
   * This function is used to read data from the stream socket. The function
   * call will block until data has been received successfully or an error
   * occurs.
   *
   * @param data The buffer into which the data will be read.
   *
   * @param max_length The maximum size of the data to be read, in bytes.
   *
   * @returns The number of bytes read or 0 if the connection was closed
   * cleanly.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note The read operation may not receive all of the requested number of
   * bytes. Consider using the asio::read_n() function if you need to ensure
   * that the requested amount of data is read before the blocking operation
   * completes.
   */
  size_t read(void* data, size_t max_length)
  {
    return service_.receive(impl_, data,
        max_length, message_flags(0), default_error_handler());
  }

  /// Read some data from the socket.
  /**
   * This function is used to read data from the stream socket. The function
   * call will block until data has been received successfully or an error
   * occurs.
   *
   * @param data The buffer into which the data will be read.
   *
   * @param max_length The maximum size of the data to be read, in bytes.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes read or 0 if the connection was closed
   * cleanly.
   *
   * @note The read operation may not receive all of the requested number of
   * bytes. Consider using the asio::read_n() function if you need to ensure
   * that the requested amount of data is read before the blocking operation
   * completes.
   */
  template <typename Error_Handler>
  size_t read(void* data, size_t max_length, Error_Handler error_handler)
  {
    return service_.receive(impl_, data,
        max_length, message_flags(0), error_handler);
  }

  /// Start an asynchronous read.
  /**
   * This function is used to asynchronously read data from the stream socket.
   * The function call always returns immediately.
   *
   * @param data The buffer into which the data will be read. Ownership of the
   * buffer is retained by the caller, which must guarantee that it is valid
   * until the handler is called.
   *
   * @param max_length The maximum size of the data to be read, in bytes.
   *
   * @param handler The handler to be called when the read operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error, // Result of operation
   *   size_t bytes_transferred  // Number of bytes received
   * ); @endcode
   *
   * @note The read operation may not read all of the requested number of bytes.
   * Consider using the asio::async_read_n() function if you need to ensure that
   * the requested amount of data is received before the asynchronous operation
   * completes.
   */
  template <typename Handler>
  void async_read(void* data, size_t max_length, Handler handler)
  {
    service_.async_receive(impl_, data, max_length, message_flags(0), handler);
  }

  /// Peek at the incoming data on the stream socket.
  /**
   * This function is used to peek at the incoming data on the stream socket,
   * without removing it from the input queue. The function call will block
   * until data has been read successfully or an error occurs.
   *
   * @param data The buffer into which the data will be read.
   *
   * @param max_length The maximum size of the data to be read, in bytes.
   *
   * @returns The number of bytes read or 0 if the connection was closed
   * cleanly.
   *
   * @throws asio::error Thrown on failure.
   */
  size_t peek(void* data, size_t max_length)
  {
    return service_.receive(impl_, data, max_length,
        message_peek, default_error_handler());
  }

  /// Peek at the incoming data on the stream socket.
  /**
   * This function is used to peek at the incoming data on the stream socket,
   * without removing it from the input queue. The function call will block
   * until data has been read successfully or an error occurs.
   *
   * @param data The buffer into which the data will be read.
   *
   * @param max_length The maximum size of the data to be read, in bytes.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes read or 0 if the connection was closed
   * cleanly.
   */
  template <typename Error_Handler>
  size_t peek(void* data, size_t max_length, Error_Handler error_handler)
  {
    return service_.peek(impl_, data, max_length,
        message_peek, error_handler);
  }

  /// Determine the amount of data that may be read without blocking.
  /**
   * This function is used to determine the amount of data, in bytes, that may
   * be read from the stream socket without blocking.
   *
   * @returns The number of bytes of data that can be read without blocking.
   *
   * @throws asio::error Thrown on failure.
   */
  size_t in_avail()
  {
    return service_.in_avail(impl_, default_error_handler());
  }

  /// Determine the amount of data that may be read without blocking.
  /**
   * This function is used to determine the amount of data, in bytes, that may
   * be read from the stream socket without blocking.
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
  size_t in_avail(Error_Handler error_handler)
  {
    return service_.in_avail(impl_, error_handler);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_STREAM_SOCKET_HPP
