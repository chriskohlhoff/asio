//
// basic_datagram_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
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
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/default_error_handler.hpp"
#include "asio/null_error_handler.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_base.hpp"

namespace asio {

/// Provides datagram-oriented socket functionality.
/**
 * The basic_datagram_socket class template provides asynchronous and blocking
 * datagram-oriented socket functionality.
 *
 * Most applications will use the asio::datagram_socket typedef.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object.
 */
template <typename Service>
class basic_datagram_socket
  : public socket_base,
    private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide socket operations.
  typedef Service service_type;

  /// The native implementation type of the datagram socket.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// A basic_datagram_socket is always the lowest layer.
  typedef basic_datagram_socket<service_type> lowest_layer_type;

  /// Construct a basic_datagram_socket without opening it.
  /**
   * This constructor creates a datagram socket without opening it. The open()
   * function must be called before data can be sent or received on the socket.
   *
   * @param d The demuxer object that the datagram socket will use to dispatch
   * handlers for any asynchronous operations performed on the socket.
   */
  explicit basic_datagram_socket(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
  }

  /// Construct a basic_datagram_socket, opening it and binding it to the given
  /// local endpoint.
  /**
   * This constructor creates a datagram socket and automatically opens it bound
   * to the specified endpoint on the local machine. The protocol used is the
   * protocol associated with the given endpoint.
   *
   * @param d The demuxer object that the datagram socket will use to dispatch
   * handlers for any asynchronous operations performed on the socket.
   *
   * @param endpoint An endpoint on the local machine to which the datagram
   * socket will be bound.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Endpoint>
  basic_datagram_socket(demuxer_type& d, const Endpoint& endpoint)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.open(impl_, endpoint.protocol(), default_error_handler());
    service_.bind(impl_, endpoint, default_error_handler());
  }

  /// Destructor.
  ~basic_datagram_socket()
  {
    service_.close(impl_, null_error_handler());
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the datagram
   * socket uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the demuxer object that datagram socket will use to
   * dispatch handlers. Ownership is not transferred to the caller.
   */
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Open the socket using the specified protocol.
  /**
   * This function opens the datagram socket so that it will use the specified
   * protocol.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Protocol>
  void open(const Protocol& protocol)
  {
    service_.open(impl_, protocol, default_error_handler());
  }

  /// Open the socket using the specified protocol.
  /**
   * This function opens the datagram socket so that it will use the specified
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
   * This function is used to close the datagram socket. Any asynchronous send
   * or receive operations will be cancelled immediately.
   *
   * A subsequent call to open() is required before the socket can again be
   * used to again perform send and receive operations.
   *
   * @throws asio::error Thrown on failure.
   */
  void close()
  {
    service_.close(impl_, default_error_handler());
  }

  /// Close the socket.
  /**
   * This function is used to close the datagram socket. Any asynchronous send
   * or receive operations will be cancelled immediately.
   *
   * A subsequent call to open() is required before the socket can again be
   * used to again perform send and receive operations.
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
   * layers. Since a basic_datagram_socket cannot contain any further layers,
   * it simply returns a reference to itself.
   *
   * @return A reference to the lowest layer in the stack of layers. Ownership
   * is not transferred to the caller.
   */
  lowest_layer_type& lowest_layer()
  {
    return *this;
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * datagram socket. This is intended to allow access to native socket
   * functionality that is not otherwise provided.
   */
  impl_type impl()
  {
    return impl_;
  }

  /// Set the underlying implementation in the native type.
  /**
   * This function is used by the acceptor implementation to set the underlying
   * implementation associated with the datagram socket.
   *
   * @param new_impl The new underlying socket implementation.
   */
  void set_impl(impl_type new_impl)
  {
    service_.assign(impl_, new_impl);
  }

  /// Bind the socket to the given local endpoint.
  /**
   * This function binds the datagram socket to the specified endpoint on the
   * local machine.
   *
   * @param endpoint An endpoint on the local machine to which the datagram
   * socket will be bound.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Endpoint>
  void bind(const Endpoint& endpoint)
  {
    service_.bind(impl_, endpoint, default_error_handler());
  }

  /// Bind the socket to the given local endpoint.
  /**
   * This function binds the datagram socket to the specified endpoint on the
   * local machine.
   *
   * @param endpoint An endpoint on the local machine to which the datagram
   * socket will be bound.
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

  /// Connect a datagram socket to the specified endpoint.
  /**
   * This function is used to connect a datagram socket to the specified remote
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
    service_.connect(impl_, peer_endpoint, default_error_handler());
  }

  /// Connect a datagram socket to the specified endpoint.
  /**
   * This function is used to connect a datagram socket to the specified remote
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
   * This function is used to asynchronously connect a datagram socket to the
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
    service_.io_control(impl_, command, default_error_handler());
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

  /// Disable sends or receives on the socket.
  /**
   * This function is used to disable send operations, receive operations, or
   * both.
   *
   * @param what Determines what types of operation will no longer be allowed.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void shutdown(shutdown_type what, Error_Handler error_handler)
  {
    service_.shutdown(impl_, what, error_handler);
  }

  /// Send some data on a connected socket.
  /**
   * This function is used to send data on the datagram socket. The function
   * call will block until the data has been sent successfully or an error
   * occurs.
   *
   * @param buffers The data to be sent on the socket.
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
  size_t send(const Const_Buffers& buffers, message_flags flags)
  {
    return service_.send(impl_, buffers, flags, default_error_handler());
  }

  /// Send some data on a connected socket.
  /**
   * This function is used to send data on the datagram socket. The function
   * call will block until the data has been sent successfully or an error
   * occurs.
   *
   * @param buffers The data to be sent on the socket.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes sent.
   *
   * @note The send operation can only be used with a connected socket. Use
   * the send_to function to send data on an unconnected datagram socket.
   */
  template <typename Const_Buffers, typename Error_Handler>
  size_t send(const Const_Buffers& buffers, message_flags flags,
      Error_Handler error_handler)
  {
    return service_.send(impl_, buffers, flags, error_handler);
  }

  /// Start an asynchronous send on a connected socket.
  /**
   * This function is used to send data on the datagram socket. The function
   * call will block until the data has been sent successfully or an error
   * occurs.
   *
   * @param buffers The data to be sent on the socket. Although the buffers
   * object may be copied as necessary, ownership of the underlying buffers is
   * retained by the caller, which must guarantee that they remain valid until
   * the handler is called.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @param handler The handler to be called when the send operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error, // Result of operation
   *   size_t bytes_transferred  // Number of bytes sent
   * ); @endcode
   *
   * @note The async_send operation can only be used with a connected socket.
   * Use the async_send_to function to send data on an unconnected datagram
   * socket.
   */
  template <typename Const_Buffers, typename Handler>
  void async_send(const Const_Buffers& buffers, message_flags flags,
      Handler handler)
  {
    service_.async_send(impl_, buffers, flags, handler);
  }

  /// Send a datagram to the specified endpoint.
  /**
   * This function is used to send a datagram to the specified remote endpoint.
   * The function call will block until the data has been sent successfully or
   * an error occurs.
   *
   * @param buffers The data to be sent to the remote endpoint.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @param destination The remote endpoint to which the data will be sent.
   *
   * @returns The number of bytes sent.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Const_Buffers, typename Endpoint>
  size_t send_to(const Const_Buffers& buffers, message_flags flags,
      const Endpoint& destination)
  {
    return service_.send_to(impl_, buffers, flags, destination,
        default_error_handler());
  }

  /// Send a datagram to the specified endpoint.
  /**
   * This function is used to send a datagram to the specified remote endpoint.
   * The function call will block until the data has been sent successfully or
   * an error occurs.
   *
   * @param buffers The data to be sent to the remote endpoint.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @param destination The remote endpoint to which the data will be sent.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes sent.
   */
  template <typename Const_Buffers, typename Endpoint, typename Error_Handler>
  size_t send_to(const Const_Buffers& buffers, message_flags flags,
      const Endpoint& destination, Error_Handler error_handler)
  {
    return service_.send_to(impl_, buffers, flags, destination, error_handler);
  }

  /// Start an asynchronous send.
  /**
   * This function is used to asynchronously send a datagram to the specified
   * remote endpoint. The function call always returns immediately.
   *
   * @param buffers The data to be sent to the remote endpoint. Although the
   * buffers object may be copied as necessary, ownership of the underlying
   * buffers is retained by the caller, which must guarantee that they remain
   * valid until the handler is called.
   *
   * @param flags Flags specifying how the send call is to be made.
   *
   * @param destination The remote endpoint to which the data will be sent.
   * Copies will be made of the endpoint as required.
   *
   * @param handler The handler to be called when the send operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error, // Result of operation
   *   size_t bytes_transferred  // Number of bytes sent
   * ); @endcode
   */
  template <typename Const_Buffers, typename Endpoint, typename Handler>
  void async_send_to(const Const_Buffers& buffers, message_flags flags,
      const Endpoint& destination, Handler handler)
  {
    service_.async_send_to(impl_, buffers, flags, destination, handler);
  }

  /// Receive some data on a connected socket.
  /**
   * This function is used to receive data on the datagram socket. The function
   * call will block until data has been received successfully or an error
   * occurs.
   *
   * @param buffers The buffers into which the data will be received.
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
  size_t receive(const Mutable_Buffers& buffers, message_flags flags)
  {
    return service_.receive(impl_, buffers, flags, default_error_handler());
  }

  /// Receive some data on a connected socket.
  /**
   * This function is used to receive data on the datagram socket. The function
   * call will block until data has been received successfully or an error
   * occurs.
   *
   * @param buffers The buffers into which the data will be received.
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
   * @returns The number of bytes received.
   *
   * @note The receive operation can only be used with a connected socket. Use
   * the receive_from function to receive data on an unconnected datagram
   * socket.
   */
  template <typename Mutable_Buffers, typename Error_Handler>
  size_t receive(const Mutable_Buffers& buffers, message_flags flags,
      Error_Handler error_handler)
  {
    return service_.receive(impl_, buffers, flags, error_handler);
  }

  /// Start an asynchronous receive on a connected socket.
  /**
   * This function is used to asynchronously receive data from the datagram
   * socket. The function call always returns immediately.
   *
   * @param data The buffers into which the data will be received. Although the
   * buffers object may be copied as necessary, ownership of the underlying
   * buffers is retained by the caller, which must guarantee that they remain
   * valid until the handler is called.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @param handler The handler to be called when the receive operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error& error, // Result of operation
   *   size_t bytes_transferred  // Number of bytes received
   * ); @endcode
   *
   * @note The async_receive operation can only be used with a connected socket.
   * Use the async_receive_from function to receive data on an unconnected
   * datagram socket.
   */
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(const Mutable_Buffers& buffers, message_flags flags,
      Handler handler)
  {
    service_.async_receive(impl_, buffers, flags, handler);
  }

  /// Receive a datagram with the endpoint of the sender.
  /**
   * This function is used to receive a datagram. The function call will block
   * until data has been received successfully or an error occurs.
   *
   * @param buffers The buffers into which the data will be received.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @param sender_endpoint An endpoint object that receives the endpoint of
   * the remote sender of the datagram.
   *
   * @returns The number of bytes received.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Mutable_Buffers, typename Endpoint>
  size_t receive_from(const Mutable_Buffers& buffers, message_flags flags,
      Endpoint& sender_endpoint)
  {
    return service_.receive_from(impl_, buffers, flags, sender_endpoint,
        default_error_handler());
  }
  
  /// Receive a datagram with the endpoint of the sender.
  /**
   * This function is used to receive a datagram. The function call will block
   * until data has been received successfully or an error occurs.
   *
   * @param buffers The buffers into which the data will be received.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @param sender_endpoint An endpoint object that receives the endpoint of
   * the remote sender of the datagram.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes received.
   */
  template <typename Mutable_Buffers, typename Endpoint, typename Error_Handler>
  size_t receive_from(const Mutable_Buffers& buffers, message_flags flags,
      Endpoint& sender_endpoint, Error_Handler error_handler)
  {
    return service_.receive_from(impl_, buffers, flags, sender_endpoint,
        error_handler);
  }
  
  /// Start an asynchronous receive.
  /**
   * This function is used to asynchronously receive a datagram. The function
   * call always returns immediately.
   *
   * @param data The buffers into which the data will be received. Although the
   * buffers object may be copied as necessary, ownership of the underlying
   * buffers is retained by the caller, which must guarantee that they remain
   * valid until the handler is called.
   *
   * @param flags Flags specifying how the receive call is to be made.
   *
   * @param sender_endpoint An endpoint object that receives the endpoint of
   * the remote sender of the datagram. Ownership of the sender_endpoint object
   * is retained by the caller, which must guarantee that it is valid until the
   * handler is called.
   *
   * @param handler The handler to be called when the receive operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error& error, // Result of operation
   *   size_t bytes_recvd        // Number of bytes received
   * ); @endcode
   */
  template <typename Mutable_Buffers, typename Endpoint, typename Handler>
  void async_receive_from(const Mutable_Buffers& buffers, message_flags flags,
      Endpoint& sender_endpoint, Handler handler)
  {
    service_.async_receive_from(impl_, buffers, flags, sender_endpoint,
        handler);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_DATAGRAM_SOCKET_HPP
