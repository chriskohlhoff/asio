//
// basic_dgram_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_BASIC_DGRAM_SOCKET_HPP
#define ASIO_BASIC_DGRAM_SOCKET_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error_handler.hpp"
#include "asio/null_completion_context.hpp"
#include "asio/service_factory.hpp"

namespace asio {

/// The basic_dgram_socket class template provides asynchronous and blocking
/// datagram-oriented socket functionality. Most applications will use the
/// dgram_socket typedef.
template <typename Service>
class basic_dgram_socket
  : private boost::noncopyable
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
   * @param d The demuxer object that the dgram socket will use to deliver
   * completions for any asynchronous operations performed on the socket.
   */
  explicit basic_dgram_socket(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
  }

  /// Construct a basic_dgram_socket opened on the given address.
  /**
   * This constructor creates a dgram socket and automatically opens it bound
   * to the specified address on the local machine.
   *
   * @param d The demuxer object that the dgram socket will use to deliver
   * completions for any asynchronous operations performed on the socket.
   *
   * @param address An address on the local machine to which the dgram socket
   * will be bound.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Address>
  basic_dgram_socket(demuxer_type& d, const Address& address)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.create(impl_, address, default_error_handler());
  }

  /// Construct a basic_dgram_socket opened on the given address.
  /**
   * This constructor creates a dgram socket and automatically opens it bound
   * to the specified address on the local machine.
   *
   * @param d The demuxer object that the dgram socket will use to deliver
   * completions for any asynchronous operations performed on the socket.
   *
   * @param address An address on the local machine to which the dgram socket
   * will be bound.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   */
  template <typename Address, typename Error_Handler>
  basic_dgram_socket(demuxer_type& d, const Address& address,
      Error_Handler error_handler)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.create(impl_, address, error_handler);
  }

  /// Destructor.
  ~basic_dgram_socket()
  {
    service_.destroy(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the dgram
   * socket uses to deliver completions for asynchronous operations.
   *
   * @return A reference to the demuxer object that dgram socket will use to
   * deliver completion notifications. Ownership is not transferred to the
   * caller.
   */
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Open the socket on the given address.
  /**
   * This function opens the dgram socket so that it is bound to the specified
   * address on the local machine.
   *
   * @param address An address on the local machine to which the dgram socket
   * will be bound.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Address>
  void open(const Address& address)
  {
    service_.create(impl_, address, default_error_handler());
  }

  /// Open the socket on the given address.
  /**
   * This function opens the dgram socket so that it is bound to the specified
   * address on the local machine.
   *
   * @param address An address on the local machine to which the dgram socket
   * will be bound.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   */
  template <typename Address, typename Error_Handler>
  void open(const Address& address, Error_Handler error_handler)
  {
    service_.create(impl_, address, error_handler);
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
    service_.destroy(impl_);
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

  /// Get the local address of the socket.
  /**
   * This function is used to obtain the locally bound address of the socket.
   *
   * @param address An address object that receives the local address of the
   * socket.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Address>
  void get_local_address(Address& address)
  {
    service_.get_local_address(impl_, address, default_error_handler());
  }

  /// Get the local address of the socket.
  /**
   * This function is used to obtain the locally bound address of the socket.
   *
   * @param address An address object that receives the local address of the
   * socket.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   */
  template <typename Address, typename Error_Handler>
  void get_local_address(Address& address, Error_Handler error_handler)
  {
    service_.get_local_address(impl_, address, error_handler);
  }

  /// Send a datagram to the specified address.
  /**
   * This function is used to send a datagram to the specified remote address.
   * The function call will block until the data has been sent successfully or
   * an error occurs.
   *
   * @param data The data to be sent to remote address.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param destination The remote address to which the data will be sent.
   *
   * @returns The number of bytes sent.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Address>
  size_t sendto(const void* data, size_t length, const Address& destination)
  {
    return service_.sendto(impl_, data, length, destination,
        default_error_handler());
  }

  /// Send a datagram to the specified address.
  /**
   * This function is used to send a datagram to the specified remote address.
   * The function call will block until the data has been sent successfully or
   * an error occurs.
   *
   * @param data The data to be sent to remote address.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param destination The remote address to which the data will be sent.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes sent.
   */
  template <typename Address, typename Error_Handler>
  size_t sendto(const void* data, size_t length, const Address& destination,
      Error_Handler error_handler)
  {
    return service_.sendto(impl_, data, length, destination, error_handler);
  }

  /// Start an asynchronous send.
  /**
   * This function is used to asynchronously send a datagram to the specified
   * remote address. The function call always returns immediately.
   *
   * @param data The data to be sent to the remote address. Ownership of the
   * data is retained by the caller, which must guarantee that it is valid
   * until the handler is called.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param destination The remote address to which the data will be sent.
   * Copies will be made of the address as required.
   *
   * @param handler The completion handler to be called when the send operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error, // Result of operation
   *   size_t bytes_sent                // Number of bytes sent
   * ); @endcode
   */
  template <typename Address, typename Handler>
  void async_sendto(const void* data, size_t length,
      const Address& destination, Handler handler)
  {
    service_.async_sendto(impl_, data, length, destination, handler,
        null_completion_context());
  }

  /// Start an asynchronous send.
  /**
   * This function is used to asynchronously send a datagram to the specified
   * remote address. The function call always returns immediately.
   *
   * @param data The data to be sent to the remote address. Ownership of the
   * data is retained by the caller, which must guarantee that it is valid
   * until the handler is called.
   *
   * @param length The size of the data to be sent, in bytes.
   *
   * @param destination The remote address to which the data will be sent.
   * Copies will be made of the address as required.
   *
   * @param handler The completion handler to be called when the send operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be <tt>void handler(const
   * socket_error& error, size_t bytes_sent)</tt>.
   *
   * @param handler The completion handler to be called when the send operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error, // Result of operation
   *   size_t bytes_sent                // Number of bytes sent
   * ); @endcode
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Copies will be made
   * of the context object as required, however all copies are equivalent.
   */
  template <typename Address, typename Handler, typename Completion_Context>
  void async_sendto(const void* data, size_t length,
      const Address& destination, Handler handler,
      Completion_Context context)
  {
    service_.async_sendto(impl_, data, length, destination, handler, context);
  }

  /// Receive a datagram with the address of the sender.
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
   * @param sender_address An address object that receives the address of the
   * remote sender of the datagram.
   *
   * @returns The number of bytes received.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Address>
  size_t recvfrom(void* data, size_t max_length, Address& sender_address)
  {
    return service_.recvfrom(impl_, data, max_length, sender_address,
        default_error_handler());
  }
  
  /// Receive a datagram with the address of the sender.
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
   * @param sender_address An address object that receives the address of the
   * remote sender of the datagram.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   *
   * @returns The number of bytes received.
   */
  template <typename Address, typename Error_Handler>
  size_t recvfrom(void* data, size_t max_length, Address& sender_address,
      Error_Handler error_handler)
  {
    return service_.recvfrom(impl_, data, max_length, sender_address,
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
   * @param sender_address An address object that receives the address of the
   * remote sender of the datagram. Ownership of the sender_address object is
   * retained by the caller, which must guarantee that it is valid until the
   * handler is called.
   *
   * @param handler The completion handler to be called when the receive
   * operation completes. Copies will be made of the handler as required. The
   * equivalent function signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error, // Result of operation
   *   size_t bytes_received            // Number of bytes received
   * ); @endcode
   */
  template <typename Address, typename Handler>
  void async_recvfrom(void* data, size_t max_length, Address& sender_address,
      Handler handler)
  {
    service_.async_recvfrom(impl_, data, max_length, sender_address, handler,
        null_completion_context());
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
   * @param sender_address An address object that receives the address of the
   * remote sender of the datagram. Ownership of the sender_address object is
   * retained by the caller, which must guarantee that it is valid until the
   * handler is called.
   *
   * @param handler The completion handler to be called when the receive
   * operation completes. Copies will be made of the handler as required. The
   * equivalent function signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error, // Result of operation
   *   size_t bytes_received            // Number of bytes received
   * ); @endcode
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Copies will be made
   * of the context object as required, however all copies are equivalent.
   */
  template <typename Address, typename Handler, typename Completion_Context>
  void async_recvfrom(void* data, size_t max_length, Address& sender_address,
      Handler handler, Completion_Context context)
  {
    service_.async_recvfrom(impl_, data, max_length, sender_address, handler,
        context);
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
