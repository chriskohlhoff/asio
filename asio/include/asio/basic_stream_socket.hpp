//
// basic_stream_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include "asio/detail/pop_options.hpp"

#include "asio/basic_socket.hpp"
#include "asio/error_handler.hpp"
#include "asio/stream_socket_service.hpp"

namespace asio {

/// Provides stream-oriented socket functionality.
/**
 * The basic_stream_socket class template provides asynchronous and blocking
 * stream-oriented socket functionality.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Read_Stream, Async_Write_Stream, Error_Source, IO_Object, Stream,
 * Sync_Read_Stream, Sync_Write_Stream.
 */
template <typename Protocol,
    typename Service = stream_socket_service<Protocol> >
class basic_stream_socket
  : public basic_socket<Protocol, Service>
{
public:
  /// The native representation of a socket.
  typedef typename Service::native_type native_type;

  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  /// Construct a basic_stream_socket without opening it.
  /**
   * This constructor creates a stream socket without opening it. The socket
   * needs to be opened and then connected or accepted before data can be sent
   * or received on it.
   *
   * @param io_service The io_service object that the stream socket will use to
   * dispatch handlers for any asynchronous operations performed on the socket.
   */
  explicit basic_stream_socket(asio::io_service& io_service)
    : basic_socket<Protocol, Service>(io_service)
  {
  }

  /// Construct and open a basic_stream_socket.
  /**
   * This constructor creates and opens a stream socket. The socket needs to be
   * connected or accepted before data can be sent or received on it.
   *
   * @param io_service The io_service object that the stream socket will use to
   * dispatch handlers for any asynchronous operations performed on the socket.
   *
   * @param protocol An object specifying protocol parameters to be used.
   *
   * @throws asio::error Thrown on failure.
   */
  basic_stream_socket(asio::io_service& io_service,
      const protocol_type& protocol)
    : basic_socket<Protocol, Service>(io_service, protocol)
  {
  }

  /// Construct a basic_stream_socket, opening it and binding it to the given
  /// local endpoint.
  /**
   * This constructor creates a stream socket and automatically opens it bound
   * to the specified endpoint on the local machine. The protocol used is the
   * protocol associated with the given endpoint.
   *
   * @param io_service The io_service object that the stream socket will use to
   * dispatch handlers for any asynchronous operations performed on the socket.
   *
   * @param endpoint An endpoint on the local machine to which the stream
   * socket will be bound.
   *
   * @throws asio::error Thrown on failure.
   */
  basic_stream_socket(asio::io_service& io_service,
      const endpoint_type& endpoint)
    : basic_socket<Protocol, Service>(io_service, endpoint)
  {
  }

  /// Construct a basic_stream_socket on an existing native socket.
  /**
   * This constructor creates a stream socket object to hold an existing native
   * socket.
   *
   * @param io_service The io_service object that the stream socket will use to
   * dispatch handlers for any asynchronous operations performed on the socket.
   *
   * @param protocol An object specifying protocol parameters to be used.
   *
   * @param native_socket The new underlying socket implementation.
   *
   * @throws asio::error Thrown on failure.
   */
  basic_stream_socket(asio::io_service& io_service,
      const protocol_type& protocol, const native_type& native_socket)
    : basic_socket<Protocol, Service>(io_service, protocol, native_socket)
  {
  }

  /// Send some data on the socket.
  /**
   * This function is used to send data on the stream socket. The function
   * call will block until one or more bytes of the data has been sent
   * successfully, or an until error occurs.
   *
   * @param buffers One or more data buffers to be sent on the socket.
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
   * @code
   * socket.send(asio::buffer(data, size));
   * @endcode
   * See the @ref buffer documentation for information on sending multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers>
  std::size_t send(const Const_Buffers& buffers)
  {
    return this->service.send(this->implementation, buffers, 0, throw_error());
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
   * @code
   * socket.send(asio::buffer(data, size), 0);
   * @endcode
   * See the @ref buffer documentation for information on sending multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers>
  std::size_t send(const Const_Buffers& buffers,
      socket_base::message_flags flags)
  {
    return this->service.send(this->implementation, buffers, flags,
        throw_error());
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
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
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
  std::size_t send(const Const_Buffers& buffers,
      socket_base::message_flags flags,
      Error_Handler error_handler)
  {
    return this->service.send(this->implementation, buffers, flags,
        error_handler);
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
   * @param handler The handler to be called when the send operation completes.
   * Copies will be made of the handler as required. The function signature of
   * the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes sent.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @note The send operation may not transmit all of the data to the peer.
   * Consider using the @ref async_write function if you need to ensure that all
   * data is written before the asynchronous operation completes.
   *
   * @par Example:
   * To send a single data buffer use the @ref buffer function as follows:
   * @code
   * socket.async_send(asio::buffer(data, size), handler);
   * @endcode
   * See the @ref buffer documentation for information on sending multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers, typename Handler>
  void async_send(const Const_Buffers& buffers, Handler handler)
  {
    this->service.async_send(this->implementation, buffers, 0, handler);
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
   * Copies will be made of the handler as required. The function signature of
   * the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes sent.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @note The send operation may not transmit all of the data to the peer.
   * Consider using the @ref async_write function if you need to ensure that all
   * data is written before the asynchronous operation completes.
   *
   * @par Example:
   * To send a single data buffer use the @ref buffer function as follows:
   * @code
   * socket.async_send(asio::buffer(data, size), 0, handler);
   * @endcode
   * See the @ref buffer documentation for information on sending multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers, typename Handler>
  void async_send(const Const_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    this->service.async_send(this->implementation, buffers, flags, handler);
  }

  /// Receive some data on the socket.
  /**
   * This function is used to receive data on the stream socket. The function
   * call will block until one or more bytes of data has been received
   * successfully, or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @returns The number of bytes received.
   *
   * @throws asio::error Thrown on failure. An error code of
   * asio::error::eof indicates that the connection was closed by the
   * peer.
   *
   * @note The receive operation may not receive all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that the
   * requested amount of data is read before the blocking operation completes.
   *
   * @par Example:
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code
   * socket.receive(asio::buffer(data, size));
   * @endcode
   * See the @ref buffer documentation for information on receiving into
   * multiple buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers>
  std::size_t receive(const Mutable_Buffers& buffers)
  {
    return this->service.receive(this->implementation, buffers, 0,
        throw_error());
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
   * @throws asio::error Thrown on failure. An error code of
   * asio::error::eof indicates that the connection was closed by the
   * peer.
   *
   * @note The receive operation may not receive all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that the
   * requested amount of data is read before the blocking operation completes.
   *
   * @par Example:
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code
   * socket.receive(asio::buffer(data, size), 0);
   * @endcode
   * See the @ref buffer documentation for information on receiving into
   * multiple buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers>
  std::size_t receive(const Mutable_Buffers& buffers,
      socket_base::message_flags flags)
  {
    return this->service.receive(this->implementation, buffers, flags,
        throw_error());
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
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
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
  std::size_t receive(const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    return this->service.receive(this->implementation, buffers, flags,
        error_handler);
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
   * @param handler The handler to be called when the receive operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes received.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @note The receive operation may not receive all of the requested number of
   * bytes. Consider using the @ref async_read function if you need to ensure
   * that the requested amount of data is received before the asynchronous
   * operation completes.
   *
   * @par Example:
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code
   * socket.async_receive(asio::buffer(data, size), handler);
   * @endcode
   * See the @ref buffer documentation for information on receiving into
   * multiple buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(const Mutable_Buffers& buffers, Handler handler)
  {
    this->service.async_receive(this->implementation, buffers, 0, handler);
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
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes received.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @note The receive operation may not receive all of the requested number of
   * bytes. Consider using the @ref async_read function if you need to ensure
   * that the requested amount of data is received before the asynchronous
   * operation completes.
   *
   * @par Example:
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code
   * socket.async_receive(asio::buffer(data, size), 0, handler);
   * @endcode
   * See the @ref buffer documentation for information on receiving into
   * multiple buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    this->service.async_receive(this->implementation, buffers, flags, handler);
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
   * @throws asio::error Thrown on failure. An error code of
   * asio::error::eof indicates that the connection was closed by the
   * peer.
   *
   * @note The write_some operation may not transmit all of the data to the
   * peer. Consider using the @ref write function if you need to ensure that
   * all data is written before the blocking operation completes.
   *
   * @par Example:
   * To write a single data buffer use the @ref buffer function as follows:
   * @code
   * socket.write_some(asio::buffer(data, size));
   * @endcode
   * See the @ref buffer documentation for information on writing multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers>
  std::size_t write_some(const Const_Buffers& buffers)
  {
    return this->service.send(this->implementation, buffers, 0, throw_error());
  }

  /// Write some data to the socket.
  /**
   * This function is used to write data to the stream socket. The function call
   * will block until one or more bytes of the data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers One or more data buffers to be written to the socket.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
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
    return this->service.send(this->implementation, buffers, 0, error_handler);
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
   * Copies will be made of the handler as required. The function signature of
   * the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes written.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @note The write operation may not transmit all of the data to the peer.
   * Consider using the @ref async_write function if you need to ensure that all
   * data is written before the asynchronous operation completes.
   *
   * @par Example:
   * To write a single data buffer use the @ref buffer function as follows:
   * @code
   * socket.async_write_some(asio::buffer(data, size), handler);
   * @endcode
   * See the @ref buffer documentation for information on writing multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers, typename Handler>
  void async_write_some(const Const_Buffers& buffers, Handler handler)
  {
    this->service.async_send(this->implementation, buffers, 0, handler);
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
   * @throws asio::error Thrown on failure. An error code of
   * asio::error::eof indicates that the connection was closed by the
   * peer.
   *
   * @note The read_some operation may not read all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that
   * the requested amount of data is read before the blocking operation
   * completes.
   *
   * @par Example:
   * To read into a single data buffer use the @ref buffer function as follows:
   * @code
   * socket.read_some(asio::buffer(data, size));
   * @endcode
   * See the @ref buffer documentation for information on reading into multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers>
  std::size_t read_some(const Mutable_Buffers& buffers)
  {
    return this->service.receive(this->implementation, buffers, 0,
        throw_error());
  }

  /// Read some data from the socket.
  /**
   * This function is used to read data from the stream socket. The function
   * call will block until one or more bytes of data has been read successfully,
   * or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
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
    return this->service.receive(this->implementation, buffers, 0,
        error_handler);
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
   * Copies will be made of the handler as required. The function signature of
   * the handler must be:
   * @code void handler(
   *   const asio::error& error,     // Result of operation.
   *   std::size_t bytes_transferred // Number of bytes read.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @note The read operation may not read all of the requested number of bytes.
   * Consider using the @ref async_read function if you need to ensure that the
   * requested amount of data is read before the asynchronous operation
   * completes.
   *
   * @par Example:
   * To read into a single data buffer use the @ref buffer function as follows:
   * @code
   * socket.async_read_some(asio::buffer(data, size), handler);
   * @endcode
   * See the @ref buffer documentation for information on reading into multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_read_some(const Mutable_Buffers& buffers, Handler handler)
  {
    this->service.async_receive(this->implementation, buffers, 0, handler);
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
    return this->service.receive(this->implementation, buffers,
        socket_base::message_peek, throw_error());
  }

  /// Peek at the incoming data on the stream socket.
  /**
   * This function is used to peek at the incoming data on the stream socket,
   * without removing it from the input queue. The function call will block
   * until data has been read successfully or an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
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
    return this->service.receive(this->implementation, buffers,
        socket_base::message_peek, error_handler);
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
    socket_base::bytes_readable command;
    this->service.io_control(this->implementation, command, throw_error());
    return command.get();
  }

  /// Determine the amount of data that may be read without blocking.
  /**
   * This function is used to determine the amount of data, in bytes, that may
   * be read from the stream socket without blocking.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes of data that can be read without blocking.
   */
  template <typename Error_Handler>
  std::size_t in_avail(Error_Handler error_handler)
  {
    socket_base::bytes_readable command;
    this->service.io_control(this->implementation, command, error_handler);
    return command.get();
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_STREAM_SOCKET_HPP
