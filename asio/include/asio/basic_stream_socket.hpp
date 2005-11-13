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
#include <cstddef>
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/error_handler.hpp"
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
 * Async_Object, Async_Read_Stream, Async_Write_Stream, Error_Source, Stream,
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

  /// The type used for reporting errors.
  typedef asio::error error_type;

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
      impl_(service_.null())
  {
  }

  /// Destructor.
  ~basic_stream_socket()
  {
    service_.close(impl_, ignore_error());
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

  /// Open the socket using the specified protocol.
  /**
   * This function opens the stream socket so that it will use the specified
   * protocol.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Protocol>
  void open(const Protocol& protocol)
  {
    service_.open(impl_, protocol, throw_error());
  }

  /// Open the socket using the specified protocol.
  /**
   * This function opens the stream socket so that it will use the specified
   * protocol.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Protocol, typename Error_Handler>
  void open(const Protocol& protocol, Error_Handler error_handler)
  {
    service_.open(impl_, protocol, error_handler);
  }

  /// Close the socket.
  /**
   * This function is used to close the stream socket. Any asynchronous send
   * or receive operations will be cancelled immediately.
   *
   * @throws asio::error Thrown on failure.
   */
  void close()
  {
    service_.close(impl_, throw_error());
  }

  /// Close the socket.
  /**
   * This function is used to close the stream socket. Any asynchronous send
   * or receive operations will be cancelled immediately.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void close(Error_Handler error_handler)
  {
    service_.close(impl_, error_handler);
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
   * This function is used by the acceptor implementation to set the underlying
   * implementation associated with the stream socket.
   *
   * @param new_impl The new underlying socket implementation.
   */
  void set_impl(impl_type new_impl)
  {
    service_.assign(impl_, new_impl);
  }

  /// Bind the socket to the given local endpoint.
  /**
   * This function binds the stream socket to the specified endpoint on the
   * local machine.
   *
   * @param endpoint An endpoint on the local machine to which the stream socket
   * will be bound.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Endpoint>
  void bind(const Endpoint& endpoint)
  {
    service_.bind(impl_, endpoint, throw_error());
  }

  /// Bind the socket to the given local endpoint.
  /**
   * This function binds the stream socket to the specified endpoint on the
   * local machine.
   *
   * @param endpoint An endpoint on the local machine to which the stream socket
   * will be bound.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Endpoint, typename Error_Handler>
  void bind(const Endpoint& endpoint, Error_Handler error_handler)
  {
    service_.bind(impl_, endpoint, error_handler);
  }

  /// Connect a stream socket to the specified endpoint.
  /**
   * This function is used to connect a stream socket to the specified remote
   * endpoint. The function call will block until the connection is successfully
   * made or an error occurs.
   *
   * @param peer_endpoint The remote endpoint to which the socket will be
   * connected.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Endpoint>
  void connect(const Endpoint& peer_endpoint)
  {
    service_.connect(impl_, peer_endpoint, throw_error());
  }

  /// Connect a stream socket to the specified endpoint.
  /**
   * This function is used to connect a stream socket to the specified remote
   * endpoint. The function call will block until the connection is successfully
   * made or an error occurs.
   *
   * @param peer_endpoint The remote endpoint to which the socket will be
   * connected.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Endpoint, typename Error_Handler>
  void connect(const Endpoint& peer_endpoint, Error_Handler error_handler)
  {
    service_.connect(impl_, peer_endpoint, error_handler);
  }

  /// Start an asynchronous connect.
  /**
   * This function is used to asynchronously connect a stream socket to the
   * specified remote endpoint. The function call always returns immediately.
   *
   * @param peer_endpoint The remote endpoint to which the socket will be
   * connected. Copies will be made of the endpoint object as required.
   *
   * @param handler The handler to be called when the connection operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Endpoint, typename Handler>
  void async_connect(const Endpoint& peer_endpoint, Handler handler)
  {
    service_.async_connect(impl_, peer_endpoint, handler);
  }

  /// Set an option on the socket.
  /**
   * This function is used to set an option on the socket.
   *
   * @param option The new option value to be set on the socket.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Socket_Option>
  void set_option(const Socket_Option& option)
  {
    service_.set_option(impl_, option, throw_error());
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
  template <typename Socket_Option, typename Error_Handler>
  void set_option(const Socket_Option& option, Error_Handler error_handler)
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
  template <typename Socket_Option>
  void get_option(Socket_Option& option) const
  {
    service_.get_option(impl_, option, throw_error());
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
  template <typename Socket_Option, typename Error_Handler>
  void get_option(Socket_Option& option, Error_Handler error_handler) const
  {
    service_.get_option(impl_, option, error_handler);
  }

  /// Perform an IO control command on the socket.
  /**
   * This function is used to execute an IO control command on the socket.
   *
   * @param command The IO control command to be performed on the socket.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename IO_Control_Command>
  void io_control(IO_Control_Command& command)
  {
    service_.io_control(impl_, command, throw_error());
  }

  /// Perform an IO control command on the socket.
  /**
   * This function is used to execute an IO control command on the socket.
   *
   * @param command The IO control command to be performed on the socket.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename IO_Control_Command, typename Error_Handler>
  void io_control(IO_Control_Command& command, Error_Handler error_handler)
  {
    service_.io_control(impl_, command, error_handler);
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
    service_.get_local_endpoint(impl_, endpoint, throw_error());
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
    service_.get_remote_endpoint(impl_, endpoint, throw_error());
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
    service_.shutdown(impl_, what, throw_error());
  }

  /// Send some data on the socket.
  /**
   * This function is used to send data on the stream socket. The function
   * call will block until one or more bytes of the data has been sent
   * successfully, or an until error occurs.
   *
   * @param buffers One or more data buffers to be sent on the socket.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @returns The number of bytes sent.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note The send operation may not transmit all of the data to the peer.
   * Consider using the @ref write function if you need to ensure that all data
   * is written before the blocking operation completes.
   *
   * @par Example:
   * To send a single data buffer use the @ref buffer function as follows:
   * @code socket.send(asio::buffer(data, size), 0); @endcode
   * See the @ref buffer documentation for information on sending multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers>
  std::size_t send(const Const_Buffers& buffers, message_flags flags)
  {
    return service_.send(impl_, buffers, flags, throw_error());
  }

  /// Send some data on the socket.
  /**
   * This function is used to send data on the stream socket. The function
   * call will block until one or more bytes of the data has been sent
   * successfully, or an until error occurs.
   *
   * @param buffers One or more data buffers to be sent on the socket.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation.
   * ); @endcode
   *
   * @returns The number of bytes sent. Returns 0 if an error occurred and the
   * error handler did not throw an exception.
   *
   * @note The send operation may not transmit all of the data to the peer.
   * Consider using the @ref write function if you need to ensure that all data
   * is written before the blocking operation completes.
   */
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t send(const Const_Buffers& buffers, message_flags flags,
      Error_Handler error_handler)
  {
    return service_.send(impl_, buffers, flags, error_handler);
  }

  /// Start an asynchronous send.
  /**
   * This function is used to asynchronously send data on the stream socket.
   * The function call always returns immediately.
   *
   * @param buffers One or more data buffers to be sent on the socket. Although
   * the buffers object may be copied as necessary, ownership of the underlying
   * memory blocks is retained by the caller, which must guarantee that they
   * remain valid until the handler is called.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @param handler The handler to be called when the send operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes sent.
   * ); @endcode
   *
   * @note The send operation may not transmit all of the data to the peer.
   * Consider using the @ref async_write function if you need to ensure that all
   * data is written before the asynchronous operation completes.
   *
   * @par Example:
   * To send a single data buffer use the @ref buffer function as follows:
   * @code socket.async_send(asio::buffer(data, size), 0, handler); @endcode
   * See the @ref buffer documentation for information on sending multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers, typename Handler>
  void async_send(const Const_Buffers& buffers, message_flags flags,
      Handler handler)
  {
    service_.async_send(impl_, buffers, flags, handler);
  }

  /// Receive some data on the socket.
  /**
   * This function is used to receive data on the stream socket. The function
   * call will block until one or more bytes of data has been received
   * successfully, or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @returns The number of bytes received.
   *
   * @throws asio::error Thrown on failure. An error code of asio::error::eof
   * indicates that the connection was closed by the peer.
   *
   * @note The receive operation may not receive all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that the
   * requested amount of data is read before the blocking operation completes.
   *
   * @par Example:
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code socket.receive(asio::buffer(data, size), 0); @endcode
   * See the @ref buffer documentation for information on receiving into
   * multiple buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers>
  std::size_t receive(const Mutable_Buffers& buffers, message_flags flags)
  {
    return service_.receive(impl_, buffers, flags, throw_error());
  }

  /// Receive some data on a connected socket.
  /**
   * This function is used to receive data on the stream socket. The function
   * call will block until one or more bytes of data has been received
   * successfully, or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes received. Returns 0 if an error occurred and
   * the error handler did not throw an exception.
   *
   * @note The receive operation may not receive all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that the
   * requested amount of data is read before the blocking operation completes.
   */
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t receive(const Mutable_Buffers& buffers, message_flags flags,
      Error_Handler error_handler)
  {
    return service_.receive(impl_, buffers, flags, error_handler);
  }

  /// Start an asynchronous receive.
  /**
   * This function is used to asynchronously receive data from the stream
   * socket. The function call always returns immediately.
   *
   * @param buffers One or more buffers into which the data will be received.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @param handler The handler to be called when the receive operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes received.
   * ); @endcode
   *
   * @note The receive operation may not receive all of the requested number of
   * bytes. Consider using the @ref async_read function if you need to ensure
   * that the requested amount of data is received before the asynchronous
   * operation completes.
   *
   * @par Example:
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code socket.async_receive(asio::buffer(data, size), 0, handler); @endcode
   * See the @ref buffer documentation for information on receiving into
   * multiple buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(const Mutable_Buffers& buffers, message_flags flags,
      Handler handler)
  {
    service_.async_receive(impl_, buffers, flags, handler);
  }

  /// Write some data to the socket.
  /**
   * This function is used to write data to the stream socket. The function call
   * will block until one or more bytes of the data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers One or more data buffers to be written to the socket.
   *
   * @returns The number of bytes written.
   *
   * @throws asio::error Thrown on failure. An error code of asio::error::eof
   * indicates that the connection was closed by the peer.
   *
   * @note The write_some operation may not transmit all of the data to the
   * peer. Consider using the @ref write function if you need to ensure that
   * all data is written before the blocking operation completes.
   *
   * @par Example:
   * To write a single data buffer use the @ref buffer function as follows:
   * @code socket.write_some(asio::buffer(data, size)); @endcode
   * See the @ref buffer documentation for information on writing multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers>
  std::size_t write_some(const Const_Buffers& buffers)
  {
    return service_.send(impl_, buffers, 0, throw_error());
  }

  /// Write some data to the socket.
  /**
   * This function is used to write data to the stream socket. The function call
   * will block until one or more bytes of the data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers One or more data buffers to be written to the socket.
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
   * peer. Consider using the @ref write function if you need to ensure that
   * all data is written before the blocking operation completes.
   */
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t write_some(const Const_Buffers& buffers,
      Error_Handler error_handler)
  {
    return service_.send(impl_, buffers, 0, error_handler);
  }

  /// Start an asynchronous write.
  /**
   * This function is used to asynchronously write data to the stream socket.
   * The function call always returns immediately.
   *
   * @param buffers One or more data buffers to be written to the socket.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param handler The handler to be called when the write operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes written.
   * ); @endcode
   *
   * @note The write operation may not transmit all of the data to the peer.
   * Consider using the @ref async_write function if you need to ensure that all
   * data is written before the asynchronous operation completes.
   *
   * @par Example:
   * To write a single data buffer use the @ref buffer function as follows:
   * @code socket.async_write_some(asio::buffer(data, size), handler); @endcode
   * See the @ref buffer documentation for information on writing multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers, typename Handler>
  void async_write_some(const Const_Buffers& buffers, Handler handler)
  {
    service_.async_send(impl_, buffers, 0, handler);
  }

  /// Read some data from the socket.
  /**
   * This function is used to read data from the stream socket. The function
   * call will block until one or more bytes of data has been read successfully,
   * or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
   *
   * @returns The number of bytes read.
   *
   * @throws asio::error Thrown on failure. An error code of asio::error::eof
   * indicates that the connection was closed by the peer.
   *
   * @note The read_some operation may not read all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that
   * the requested amount of data is read before the blocking operation
   * completes.
   *
   * @par Example:
   * To read into a single data buffer use the @ref buffer function as follows:
   * @code socket.read_some(asio::buffer(data, size)); @endcode
   * See the @ref buffer documentation for information on reading into multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers>
  std::size_t read_some(const Mutable_Buffers& buffers)
  {
    return service_.receive(impl_, buffers, 0, throw_error());
  }

  /// Read some data from the socket.
  /**
   * This function is used to read data from the stream socket. The function
   * call will block until one or more bytes of data has been read successfully,
   * or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
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
   * bytes. Consider using the @ref read function if you need to ensure that
   * the requested amount of data is read before the blocking operation
   * completes.
   */
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t read_some(const Mutable_Buffers& buffers,
      Error_Handler error_handler)
  {
    return service_.receive(impl_, buffers, 0, error_handler);
  }

  /// Start an asynchronous read.
  /**
   * This function is used to asynchronously read data from the stream socket.
   * The function call always returns immediately.
   *
   * @param buffers One or more buffers into which the data will be read.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param handler The handler to be called when the read operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes read.
   * ); @endcode
   *
   * @note The read operation may not read all of the requested number of bytes.
   * Consider using the @ref async_read function if you need to ensure that the
   * requested amount of data is read before the asynchronous operation
   * completes.
   *
   * @par Example:
   * To read into a single data buffer use the @ref buffer function as follows:
   * @code socket.async_read(asio::buffer(data, size), handler); @endcode
   * See the @ref buffer documentation for information on reading into multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_read_some(const Mutable_Buffers& buffers, Handler handler)
  {
    service_.async_receive(impl_, buffers, 0, handler);
  }

  /// Peek at the incoming data on the stream socket.
  /**
   * This function is used to peek at the incoming data on the stream socket,
   * without removing it from the input queue. The function call will block
   * until data has been read successfully or an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
   *
   * @returns The number of bytes read.
   *
   * @throws asio::error Thrown on failure.
   *
   * @par Example:
   * To peek using a single data buffer use the @ref buffer function as
   * follows:
   * @code socket.peek(asio::buffer(data, size)); @endcode
   * See the @ref buffer documentation for information on using multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers>
  std::size_t peek(const Mutable_Buffers& buffers)
  {
    return service_.receive(impl_, buffers, message_peek, throw_error());
  }

  /// Peek at the incoming data on the stream socket.
  /**
   * This function is used to peek at the incoming data on the stream socket,
   * without removing it from the input queue. The function call will block
   * until data has been read successfully or an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
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
    return service_.receive(impl_, buffers, message_peek, error_handler);
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
  std::size_t in_avail()
  {
    bytes_readable command;
    service_.io_control(impl_, command, throw_error());
    return command.get();
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
  std::size_t in_avail(Error_Handler error_handler)
  {
    bytes_readable command;
    service_.io_control(impl_, command, error_handler);
    return command.get();
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
