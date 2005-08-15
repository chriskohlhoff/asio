//
// basic_dgram_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_DGRAM_SOCKET_HPP
#define ASIO_BASIC_DGRAM_SOCKET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/default_error_handler.hpp"
#include "asio/dgram_socket_base.hpp"
#include "asio/service_factory.hpp"

namespace asio {

/// Provides datagram-oriented socket functionality.
/**
 * The basic_dgram_socket class template provides asynchronous and blocking
 * datagram-oriented socket functionality.
 *
 * Most applications will use the asio::dgram_socket typedef.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object.
 */
template <typename Service>
class basic_dgram_socket
  : public dgram_socket_base,
    private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide socket operations.
  typedef Service service_type;

  /// The native implementation type of the dgram socket.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// Construct a basic_dgram_socket without opening it.
  /**
   * This constructor creates a dgram socket without opening it. The open()
   * function must be called before data can be sent or received on the socket.
   *
   * @param d The demuxer object that the dgram socket will use to dispatch
   * handlers for any asynchronous operations performed on the socket.
   */
  explicit basic_dgram_socket(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
  }

  /// Construct a basic_dgram_socket, opening it and binding it to the given
  /// local endpoint.
  /**
   * This constructor creates a dgram socket and automatically opens it bound
   * to the specified endpoint on the local machine. The protocol used is the
   * protocol associated with the given endpoint.
   *
   * @param d The demuxer object that the dgram socket will use to dispatch
   * handlers for any asynchronous operations performed on the socket.
   *
   * @param endpoint An endpoint on the local machine to which the dgram socket
   * will be bound.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Endpoint>
  basic_dgram_socket(demuxer_type& d, const Endpoint& endpoint)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.open(impl_, endpoint.protocol(), default_error_handler());
    service_.bind(impl_, endpoint, default_error_handler());
  }

  /// Destructor.
  ~basic_dgram_socket()
  {
    service_.close(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the dgram
   * socket uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the demuxer object that dgram socket will use to
   * dispatch handlers. Ownership is not transferred to the caller.
   */
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Open the socket using the specified protocol.
  /**
   * This function opens the dgram socket so that it will use the specified
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
   * This function opens the dgram socket so that it will use the specified
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

  /// Bind the socket to the given local endpoint.
  /**
   * This function binds the dgram socket to the specified endpoint on the
   * local machine.
   *
   * @param endpoint An endpoint on the local machine to which the dgram socket
   * will be bound.
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
   * This function binds the dgram socket to the specified endpoint on the
   * local machine.
   *
   * @param endpoint An endpoint on the local machine to which the dgram socket
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

  /// Close the socket.
  /**
   * This function is used to close the dgram socket. Any asynchronous sendto
   * or recvfrom operations will be cancelled immediately.
   *
   * A subsequent call to open() is required before the socket can again be
   * used to again perform send and receive operations.
   */
  void close()
  {
    service_.close(impl_);
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * dgram socket. This is intended to allow access to native socket
   * functionality that is not otherwise provided.
   */
  impl_type impl()
  {
    return impl_;
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
   *   const asio::error& error // Result of operation
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
   * @throws asio::error Thrown on failure.
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
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(Endpoint& endpoint, Error_Handler error_handler)
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

  /// Send a datagram to the specified endpoint.
  /**
   * This function is used to send a datagram to the specified remote endpoint.
   * The function call will block until the data has been sent successfully or
   * an error occurs.
   *
   * @param data The data to be sent to remote endpoint.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param destination The remote endpoint to which the data will be sent.
   *
   * @returns The number of bytes sent.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Endpoint>
  size_t sendto(const void* data, size_t length, const Endpoint& destination)
  {
    return service_.sendto(impl_, data, length, destination,
        default_error_handler());
  }

  /// Send a datagram to the specified endpoint.
  /**
   * This function is used to send a datagram to the specified remote endpoint.
   * The function call will block until the data has been sent successfully or
   * an error occurs.
   *
   * @param data The data to be sent to remote endpoint.
   *
   * @param length The size of the data to be sent, in bytes.
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
  template <typename Endpoint, typename Error_Handler>
  size_t sendto(const void* data, size_t length, const Endpoint& destination,
      Error_Handler error_handler)
  {
    return service_.sendto(impl_, data, length, destination, error_handler);
  }

  /// Start an asynchronous send.
  /**
   * This function is used to asynchronously send a datagram to the specified
   * remote endpoint. The function call always returns immediately.
   *
   * @param data The data to be sent to the remote endpoint. Ownership of the
   * data is retained by the caller, which must guarantee that it is valid
   * until the handler is called.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param destination The remote endpoint to which the data will be sent.
   * Copies will be made of the endpoint as required.
   *
   * @param handler The handler to be called when the send operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error, // Result of operation
   *   size_t bytes_sent         // Number of bytes sent
   * ); @endcode
   */
  template <typename Endpoint, typename Handler>
  void async_sendto(const void* data, size_t length,
      const Endpoint& destination, Handler handler)
  {
    service_.async_sendto(impl_, data, length, destination, handler);
  }

  /// Receive a datagram with the endpoint of the sender.
  /**
   * This function is used to receive a datagram. The function call will block
   * until data has been received successfully or an error occurs.
   *
   * @param data The data buffer into which the received datagram will be
   * written.
   *
   * @param max_length The maximum length, in bytes, of data that can be held
   * in the supplied buffer.
   *
   * @param sender_endpoint An endpoint object that receives the endpoint of
   * the remote sender of the datagram.
   *
   * @returns The number of bytes received.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Endpoint>
  size_t recvfrom(void* data, size_t max_length, Endpoint& sender_endpoint)
  {
    return service_.recvfrom(impl_, data, max_length, sender_endpoint,
        default_error_handler());
  }
  
  /// Receive a datagram with the endpoint of the sender.
  /**
   * This function is used to receive a datagram. The function call will block
   * until data has been received successfully or an error occurs.
   *
   * @param data The data buffer into which the received datagram will be
   * written.
   *
   * @param max_length The maximum length, in bytes, of data that can be held
   * in the supplied buffer.
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
  template <typename Endpoint, typename Error_Handler>
  size_t recvfrom(void* data, size_t max_length, Endpoint& sender_endpoint,
      Error_Handler error_handler)
  {
    return service_.recvfrom(impl_, data, max_length, sender_endpoint,
        error_handler);
  }
  
  /// Start an asynchronous receive.
  /**
   * This function is used to asynchronously receive a datagram. The function
   * call always returns immediately.
   *
   * @param data The data buffer into which the received datagram will be
   * written. Ownership of the data buffer is retained by the caller, which
   * must guarantee that it is valid until the handler is called.
   *
   * @param max_length The maximum length, in bytes, of data that can be held
   * in the supplied buffer.
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
  template <typename Endpoint, typename Handler>
  void async_recvfrom(void* data, size_t max_length, Endpoint& sender_endpoint,
      Handler handler)
  {
    service_.async_recvfrom(impl_, data, max_length, sender_endpoint, handler);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_DGRAM_SOCKET_HPP
