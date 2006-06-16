//
// basic_datagram_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_DATAGRAM_SOCKET_HPP
#define ASIO_BASIC_DATAGRAM_SOCKET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_socket.hpp"
#include "asio/datagram_socket_service.hpp"
#include "asio/error_handler.hpp"

namespace asio {

/// Provides datagram-oriented socket functionality.
/**
 * The basic_datagram_socket class template provides asynchronous and blocking
 * datagram-oriented socket functionality.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object, Error_Source.
 */
template <typename Protocol,
    typename Service = datagram_socket_service<Protocol> >
class basic_datagram_socket
  : public basic_socket<Protocol, Service>
{
public:
  /// The native representation of a socket.
  typedef typename Service::native_type native_type;

  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  /// Construct a basic_datagram_socket without opening it.
  /**
   * This constructor creates a datagram socket without opening it. The open()
   * function must be called before data can be sent or received on the socket.
   *
   * @param io_service The io_service object that the datagram socket will use
   * to dispatch handlers for any asynchronous operations performed on the
   * socket.
   */
  explicit basic_datagram_socket(asio::io_service& io_service)
    : basic_socket<Protocol, Service>(io_service)
  {
  }

  /// Construct and open a basic_datagram_socket.
  /**
   * This constructor creates and opens a datagram socket.
   *
   * @param io_service The io_service object that the datagram socket will use
   * to dispatch handlers for any asynchronous operations performed on the
   * socket.
   *
   * @param protocol An object specifying protocol parameters to be used.
   *
   * @throws asio::error Thrown on failure.
   */
  basic_datagram_socket(asio::io_service& io_service,
      const protocol_type& protocol)
    : basic_socket<Protocol, Service>(io_service, protocol)
  {
  }

  /// Construct a basic_datagram_socket, opening it and binding it to the given
  /// local endpoint.
  /**
   * This constructor creates a datagram socket and automatically opens it bound
   * to the specified endpoint on the local machine. The protocol used is the
   * protocol associated with the given endpoint.
   *
   * @param io_service The io_service object that the datagram socket will use
   * to dispatch handlers for any asynchronous operations performed on the
   * socket.
   *
   * @param endpoint An endpoint on the local machine to which the datagram
   * socket will be bound.
   *
   * @throws asio::error Thrown on failure.
   */
  basic_datagram_socket(asio::io_service& io_service,
      const endpoint_type& endpoint)
    : basic_socket<Protocol, Service>(io_service, endpoint)
  {
  }

  /// Construct a basic_datagram_socket on an existing native socket.
  /**
   * This constructor creates a datagram socket object to hold an existing
   * native socket.
   *
   * @param io_service The io_service object that the datagram socket will use
   * to dispatch handlers for any asynchronous operations performed on the
   * socket.
   *
   * @param protocol An object specifying protocol parameters to be used.
   *
   * @param native_socket The new underlying socket implementation.
   *
   * @throws asio::error Thrown on failure.
   */
  basic_datagram_socket(asio::io_service& io_service,
      const protocol_type& protocol, const native_type& native_socket)
    : basic_socket<Protocol, Service>(io_service, protocol, native_socket)
  {
  }

  /// Send some data on a connected socket.
  /**
   * This function is used to send data on the datagram socket. The function
   * call will block until the data has been sent successfully or an error
   * occurs.
   *
   * @param buffers One ore more data buffers to be sent on the socket.
   *
   * @returns The number of bytes sent.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note The send operation can only be used with a connected socket. Use
   * the send_to function to send data on an unconnected datagram socket.
   *
   * @par Example:
   * To send a single data buffer use the @ref buffer function as follows:
   * @code socket.send(asio::buffer(data, size)); @endcode
   * See the @ref buffer documentation for information on sending multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers>
  std::size_t send(const Const_Buffers& buffers)
  {
    return this->service.send(this->implementation, buffers, 0, throw_error());
  }

  /// Send some data on a connected socket.
  /**
   * This function is used to send data on the datagram socket. The function
   * call will block until the data has been sent successfully or an error
   * occurs.
   *
   * @param buffers One ore more data buffers to be sent on the socket.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @returns The number of bytes sent.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note The send operation can only be used with a connected socket. Use
   * the send_to function to send data on an unconnected datagram socket.
   */
  template <typename Const_Buffers>
  std::size_t send(const Const_Buffers& buffers,
      socket_base::message_flags flags)
  {
    return this->service.send(this->implementation, buffers, flags,
        throw_error());
  }

  /// Send some data on a connected socket.
  /**
   * This function is used to send data on the datagram socket. The function
   * call will block until the data has been sent successfully or an error
   * occurs.
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
   * @returns The number of bytes sent.
   *
   * @note The send operation can only be used with a connected socket. Use
   * the send_to function to send data on an unconnected datagram socket.
   */
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t send(const Const_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    return this->service.send(this->implementation, buffers, flags,
        error_handler);
  }

  /// Start an asynchronous send on a connected socket.
  /**
   * This function is used to send data on the datagram socket. The function
   * call will block until the data has been sent successfully or an error
   * occurs.
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
   * @note The async_send operation can only be used with a connected socket.
   * Use the async_send_to function to send data on an unconnected datagram
   * socket.
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

  /// Start an asynchronous send on a connected socket.
  /**
   * This function is used to send data on the datagram socket. The function
   * call will block until the data has been sent successfully or an error
   * occurs.
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
   * @note The async_send operation can only be used with a connected socket.
   * Use the async_send_to function to send data on an unconnected datagram
   * socket.
   */
  template <typename Const_Buffers, typename Handler>
  void async_send(const Const_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    this->service.async_send(this->implementation, buffers, flags, handler);
  }

  /// Send a datagram to the specified endpoint.
  /**
   * This function is used to send a datagram to the specified remote endpoint.
   * The function call will block until the data has been sent successfully or
   * an error occurs.
   *
   * @param buffers One or more data buffers to be sent to the remote endpoint.
   *
   * @param destination The remote endpoint to which the data will be sent.
   *
   * @returns The number of bytes sent.
   *
   * @throws asio::error Thrown on failure.
   *
   * @par Example:
   * To send a single data buffer use the @ref buffer function as follows:
   * @code
   * asio::ip::udp::endpoint destination(
   *     asio::ip::address::from_string("1.2.3.4"), 12345);
   * socket.send_to(asio::buffer(data, size), destination);
   * @endcode
   * See the @ref buffer documentation for information on sending multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers>
  std::size_t send_to(const Const_Buffers& buffers,
      const endpoint_type& destination)
  {
    return this->service.send_to(this->implementation, buffers, destination, 0,
        throw_error());
  }

  /// Send a datagram to the specified endpoint.
  /**
   * This function is used to send a datagram to the specified remote endpoint.
   * The function call will block until the data has been sent successfully or
   * an error occurs.
   *
   * @param buffers One or more data buffers to be sent to the remote endpoint.
   *
   * @param destination The remote endpoint to which the data will be sent.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @returns The number of bytes sent.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Const_Buffers>
  std::size_t send_to(const Const_Buffers& buffers,
      const endpoint_type& destination, socket_base::message_flags flags)
  {
    return this->service.send_to(this->implementation, buffers, destination,
        flags, throw_error());
  }

  /// Send a datagram to the specified endpoint.
  /**
   * This function is used to send a datagram to the specified remote endpoint.
   * The function call will block until the data has been sent successfully or
   * an error occurs.
   *
   * @param buffers One or more data buffers to be sent to the remote endpoint.
   *
   * @param destination The remote endpoint to which the data will be sent.
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
   * @returns The number of bytes sent.
   */
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t send_to(const Const_Buffers& buffers,
      const endpoint_type& destination, socket_base::message_flags flags,
      Error_Handler error_handler)
  {
    return this->service.send_to(this->implementation, buffers, destination,
        flags, error_handler);
  }

  /// Start an asynchronous send.
  /**
   * This function is used to asynchronously send a datagram to the specified
   * remote endpoint. The function call always returns immediately.
   *
   * @param buffers One or more data buffers to be sent to the remote endpoint.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param destination The remote endpoint to which the data will be sent.
   * Copies will be made of the endpoint as required.
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
   * @par Example:
   * To send a single data buffer use the @ref buffer function as follows:
   * @code
   * asio::ip::udp::endpoint destination(
   *     asio::ip::address::from_string("1.2.3.4"), 12345);
   * socket.async_send_to(
   *     asio::buffer(data, size), destination, handler);
   * @endcode
   * See the @ref buffer documentation for information on sending multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Const_Buffers, typename Handler>
  void async_send_to(const Const_Buffers& buffers,
      const endpoint_type& destination, Handler handler)
  {
    this->service.async_send_to(this->implementation, buffers, destination, 0,
        handler);
  }

  /// Start an asynchronous send.
  /**
   * This function is used to asynchronously send a datagram to the specified
   * remote endpoint. The function call always returns immediately.
   *
   * @param buffers One or more data buffers to be sent to the remote endpoint.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @param destination The remote endpoint to which the data will be sent.
   * Copies will be made of the endpoint as required.
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
   */
  template <typename Const_Buffers, typename Handler>
  void async_send_to(const Const_Buffers& buffers,
      const endpoint_type& destination, socket_base::message_flags flags,
      Handler handler)
  {
    this->service.async_send_to(this->implementation, buffers, destination,
        flags, handler);
  }

  /// Receive some data on a connected socket.
  /**
   * This function is used to receive data on the datagram socket. The function
   * call will block until data has been received successfully or an error
   * occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @returns The number of bytes received.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note The receive operation can only be used with a connected socket. Use
   * the receive_from function to receive data on an unconnected datagram
   * socket.
   *
   * @par Example:
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code socket.receive(asio::buffer(data, size)); @endcode
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

  /// Receive some data on a connected socket.
  /**
   * This function is used to receive data on the datagram socket. The function
   * call will block until data has been received successfully or an error
   * occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @returns The number of bytes received.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note The receive operation can only be used with a connected socket. Use
   * the receive_from function to receive data on an unconnected datagram
   * socket.
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
   * This function is used to receive data on the datagram socket. The function
   * call will block until data has been received successfully or an error
   * occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation.
   * ); @endcode
   *
   * @returns The number of bytes received.
   *
   * @note The receive operation can only be used with a connected socket. Use
   * the receive_from function to receive data on an unconnected datagram
   * socket.
   */
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t receive(const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    return this->service.receive(this->implementation, buffers, flags,
        error_handler);
  }

  /// Start an asynchronous receive on a connected socket.
  /**
   * This function is used to asynchronously receive data from the datagram
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
   * @note The async_receive operation can only be used with a connected socket.
   * Use the async_receive_from function to receive data on an unconnected
   * datagram socket.
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

  /// Start an asynchronous receive on a connected socket.
  /**
   * This function is used to asynchronously receive data from the datagram
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
   * @note The async_receive operation can only be used with a connected socket.
   * Use the async_receive_from function to receive data on an unconnected
   * datagram socket.
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    this->service.async_receive(this->implementation, buffers, flags, handler);
  }

  /// Receive a datagram with the endpoint of the sender.
  /**
   * This function is used to receive a datagram. The function call will block
   * until data has been received successfully or an error occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @param sender_endpoint An endpoint object that receives the endpoint of
   * the remote sender of the datagram.
   *
   * @returns The number of bytes received.
   *
   * @throws asio::error Thrown on failure.
   *
   * @par Example:
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code
   * asio::ip::udp::endpoint sender_endpoint;
   * socket.receive_from(
   *     asio::buffer(data, size), sender_endpoint);
   * @endcode
   * See the @ref buffer documentation for information on receiving into
   * multiple buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers>
  std::size_t receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint)
  {
    return this->service.receive_from(this->implementation, buffers,
        sender_endpoint, 0, throw_error());
  }
  
  /// Receive a datagram with the endpoint of the sender.
  /**
   * This function is used to receive a datagram. The function call will block
   * until data has been received successfully or an error occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @param sender_endpoint An endpoint object that receives the endpoint of
   * the remote sender of the datagram.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @returns The number of bytes received.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Mutable_Buffers>
  std::size_t receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint, socket_base::message_flags flags)
  {
    return this->service.receive_from(this->implementation, buffers,
        sender_endpoint, flags, throw_error());
  }
  
  /// Receive a datagram with the endpoint of the sender.
  /**
   * This function is used to receive a datagram. The function call will block
   * until data has been received successfully or an error occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @param sender_endpoint An endpoint object that receives the endpoint of
   * the remote sender of the datagram.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation.
   * ); @endcode
   *
   * @returns The number of bytes received.
   */
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint, socket_base::message_flags flags,
      Error_Handler error_handler)
  {
    return this->service.receive_from(this->implementation, buffers,
        sender_endpoint, flags, error_handler);
  }

  /// Start an asynchronous receive.
  /**
   * This function is used to asynchronously receive a datagram. The function
   * call always returns immediately.
   *
   * @param buffers One or more buffers into which the data will be received.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param sender_endpoint An endpoint object that receives the endpoint of
   * the remote sender of the datagram. Ownership of the sender_endpoint object
   * is retained by the caller, which must guarantee that it is valid until the
   * handler is called.
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
   * @par Example:
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code socket.async_receive_from(
   *     asio::buffer(data, size), 0, sender_endpoint, handler); @endcode
   * See the @ref buffer documentation for information on receiving into
   * multiple buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint, Handler handler)
  {
    this->service.async_receive_from(this->implementation, buffers,
        sender_endpoint, 0, handler);
  }

  /// Start an asynchronous receive.
  /**
   * This function is used to asynchronously receive a datagram. The function
   * call always returns immediately.
   *
   * @param buffers One or more buffers into which the data will be received.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param sender_endpoint An endpoint object that receives the endpoint of
   * the remote sender of the datagram. Ownership of the sender_endpoint object
   * is retained by the caller, which must guarantee that it is valid until the
   * handler is called.
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
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint, socket_base::message_flags flags,
      Handler handler)
  {
    this->service.async_receive_from(this->implementation, buffers,
        sender_endpoint, flags, handler);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_DATAGRAM_SOCKET_HPP
