//
// basic_stream_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_BASIC_STREAM_SOCKET_HPP
#define ASIO_BASIC_STREAM_SOCKET_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/default_error_handler.hpp"
#include "asio/service_factory.hpp"

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
 * Async_Object, Async_Recv_Stream, Async_Send_Stream, Stream,
 * Sync_Recv_Stream, Sync_Send_Stream.
 */
template <typename Service>
class basic_stream_socket
  : private boost::noncopyable
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
   * This function is used to close the stream socket. Any asynchronous send
   * or recv operations will be cancelled immediately.
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
   * @throws socket_error Thrown on failure.
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
   *   const asio::socket_error& error // Result of operation
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
   * @throws socket_error Thrown on failure.
   */
  template <typename Option>
  void get_option(Option& option)
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
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   */
  template <typename Option, typename Error_Handler>
  void get_option(Option& option, Error_Handler error_handler)
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
   * @throws socket_error Thrown on failure.
   */
  template <typename Endpoint>
  void get_local_endpoint(Endpoint& endpoint)
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
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   */
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(Endpoint& endpoint, Error_Handler error_handler)
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
   * @throws socket_error Thrown on failure.
   */
  template <typename Endpoint>
  void get_remote_endpoint(Endpoint& endpoint)
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
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   */
  template <typename Endpoint, typename Error_Handler>
  void get_remote_endpoint(Endpoint& endpoint, Error_Handler error_handler)
  {
    service_.get_remote_endpoint(impl_, endpoint, error_handler);
  }

  /// Send the given data to the peer.
  /**
   * This function is used to send data to the stream socket's peer. The
   * function call will block until the data has been sent successfully or an
   * error occurs.
   *
   * @param data The data to be sent to remote peer.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @returns The number of bytes sent or 0 if the connection was closed
   * cleanly.
   *
   * @throws socket_error Thrown on failure.
   *
   * @note The send operation may not transmit all of the data to the peer.
   * Consider using the asio::send_n() function if you need to ensure that all
   * data is sent before the blocking operation completes.
   */
  size_t send(const void* data, size_t length)
  {
    return service_.send(impl_, data, length, default_error_handler());
  }

  /// Send the given data to the peer.
  /**
   * This function is used to send data to the stream socket's peer. The
   * function call will block until the data has been sent successfully or an
   * error occurs.
   *
   * @param data The data to be sent to remote peer.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes sent or 0 if the connection was closed
   * cleanly.
   *
   * @note The send operation may not transmit all of the data to the peer.
   * Consider using the asio::send_n() function if you need to ensure that all
   * data is sent before the blocking operation completes.
   */
  template <typename Error_Handler>
  size_t send(const void* data, size_t length, Error_Handler error_handler)
  {
    return service_.send(impl_, data, length, error_handler);
  }

  /// Start an asynchronous send.
  /**
   * This function is used to asynchronously send data to the stream socket's
   * peer. The function call always returns immediately.
   *
   * @param data The data to be sent to the remote peer. Ownership of the data
   * is retained by the caller, which must guarantee that it is valid until the
   * handler is called.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param handler The handler to be called when the send operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error, // Result of operation
   *   size_t bytes_sent                // Number of bytes sent
   * ); @endcode
   *
   * @note The send operation may not transmit all of the data to the peer.
   * Consider using the asio::async_send_n() function if you need to ensure
   * that all data is sent before the asynchronous operation completes.
   */
  template <typename Handler>
  void async_send(const void* data, size_t length, Handler handler)
  {
    service_.async_send(impl_, data, length, handler);
  }

  /// Receive some data from the peer.
  /**
   * This function is used to receive data from the stream socket's peer. The
   * function call will block until data has received successfully or an error
   * occurs.
   *
   * @param data The buffer into which the received data will be written.
   *
   * @param max_length The maximum size of the data to be received, in bytes.
   *
   * @returns The number of bytes received or 0 if the connection was closed
   * cleanly.
   *
   * @throws socket_error Thrown on failure.
   *
   * @note The recv operation may not receive all of the requested number of
   * bytes. Consider using the asio::recv_n() function if you need to ensure
   * that the requested amount of data is received before the blocking
   * operation completes.
   */
  size_t recv(void* data, size_t max_length)
  {
    return service_.recv(impl_, data, max_length, default_error_handler());
  }

  /// Receive some data from the peer.
  /**
   * This function is used to receive data from the stream socket's peer. The
   * function call will block until data has received successfully or an error
   * occurs.
   *
   * @param data The buffer into which the received data will be written.
   *
   * @param max_length The maximum size of the data to be received, in bytes.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes received or 0 if the connection was closed
   * cleanly.
   *
   * @note The recv operation may not receive all of the requested number of
   * bytes. Consider using the asio::recv_n() function if you need to ensure
   * that the requested amount of data is received before the blocking
   * operation completes.
   */
  template <typename Error_Handler>
  size_t recv(void* data, size_t max_length, Error_Handler error_handler)
  {
    return service_.recv(impl_, data, max_length, error_handler);
  }

  /// Start an asynchronous receive.
  /**
   * This function is used to asynchronously receive data from the stream
   * socket's peer. The function call always returns immediately.
   *
   * @param data The buffer into which the received data will be written.
   * Ownership of the buffer is retained by the caller, which must guarantee
   * that it is valid until the handler is called.
   *
   * @param max_length The maximum size of the data to be received, in bytes.
   *
   * @param handler The handler to be called when the receive operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error, // Result of operation
   *   size_t bytes_recvd               // Number of bytes received
   * ); @endcode
   *
   * @note The recv operation may not receive all of the requested number of
   * bytes. Consider using the asio::async_recv_n() function if you need to
   * ensure that the requested amount of data is received before the
   * asynchronous operation completes.
   */
  template <typename Handler>
  void async_recv(void* data, size_t max_length, Handler handler)
  {
    service_.async_recv(impl_, data, max_length, handler);
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
