//
// basic_socket.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_SOCKET_HPP
#define ASIO_BASIC_SOCKET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_io_object.hpp"
#include "asio/error.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/throw_error.hpp"

namespace asio {

/// Provides socket functionality.
/**
 * The basic_socket class template provides functionality that is common to both
 * stream-oriented and datagram-oriented sockets.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * IO_Object.
 */
template <typename Protocol, typename Service>
class basic_socket
  : public basic_io_object<Service>,
    public socket_base
{
public:
  /// The native representation of a socket.
  typedef typename Service::native_type native_type;

  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  /// A basic_socket is always the lowest layer.
  typedef basic_socket<Protocol, Service> lowest_layer_type;

  /// Construct a basic_socket without opening it.
  /**
   * This constructor creates a socket without opening it.
   *
   * @param io_service The io_service object that the socket will use to
   * dispatch handlers for any asynchronous operations performed on the socket.
   */
  explicit basic_socket(asio::io_service& io_service)
    : basic_io_object<Service>(io_service)
  {
  }

  /// Construct and open a basic_socket.
  /**
   * This constructor creates and opens a socket.
   *
   * @param io_service The io_service object that the socket will use to
   * dispatch handlers for any asynchronous operations performed on the socket.
   *
   * @param protocol An object specifying protocol parameters to be used.
   *
   * @throws asio::system_error Thrown on failure.
   */
  basic_socket(asio::io_service& io_service,
      const protocol_type& protocol)
    : basic_io_object<Service>(io_service)
  {
    asio::error_code ec;
    this->service.open(this->implementation, protocol, ec);
    asio::detail::throw_error(ec);
  }

  /// Construct a basic_socket, opening it and binding it to the given local
  /// endpoint.
  /**
   * This constructor creates a socket and automatically opens it bound to the
   * specified endpoint on the local machine. The protocol used is the protocol
   * associated with the given endpoint.
   *
   * @param io_service The io_service object that the socket will use to
   * dispatch handlers for any asynchronous operations performed on the socket.
   *
   * @param endpoint An endpoint on the local machine to which the socket will
   * be bound.
   *
   * @throws asio::system_error Thrown on failure.
   */
  basic_socket(asio::io_service& io_service,
      const endpoint_type& endpoint)
    : basic_io_object<Service>(io_service)
  {
    asio::error_code ec;
    this->service.open(this->implementation, endpoint.protocol(), ec);
    asio::detail::throw_error(ec);
    this->service.bind(this->implementation, endpoint, ec);
    asio::detail::throw_error(ec);
  }

  /// Construct a basic_socket on an existing native socket.
  /**
   * This constructor creates a socket object to hold an existing native socket.
   *
   * @param io_service The io_service object that the socket will use to
   * dispatch handlers for any asynchronous operations performed on the socket.
   *
   * @param protocol An object specifying protocol parameters to be used.
   *
   * @param native_socket A native socket.
   *
   * @throws asio::system_error Thrown on failure.
   */
  basic_socket(asio::io_service& io_service,
      const protocol_type& protocol, const native_type& native_socket)
    : basic_io_object<Service>(io_service)
  {
    asio::error_code ec;
    this->service.assign(this->implementation, protocol, native_socket, ec);
    asio::detail::throw_error(ec);
  }

  /// Get a reference to the lowest layer.
  /**
   * This function returns a reference to the lowest layer in a stack of
   * layers. Since a basic_socket cannot contain any further layers, it simply
   * returns a reference to itself.
   *
   * @return A reference to the lowest layer in the stack of layers. Ownership
   * is not transferred to the caller.
   */
  lowest_layer_type& lowest_layer()
  {
    return *this;
  }

  /// Open the socket using the specified protocol.
  /**
   * This function opens the socket so that it will use the specified protocol.
   *
   * @param protocol An object specifying protocol parameters to be used.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * socket.open(asio::ip::tcp::v4());
   * @endcode
   */
  void open(const protocol_type& protocol = protocol_type())
  {
    asio::error_code ec;
    this->service.open(this->implementation, protocol, ec);
    asio::detail::throw_error(ec);
  }

  /// Open the socket using the specified protocol.
  /**
   * This function opens the socket so that it will use the specified protocol.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * asio::error_code ec;
   * socket.open(asio::ip::tcp::v4(), ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  asio::error_code open(const protocol_type& protocol,
      asio::error_code& ec)
  {
    return this->service.open(this->implementation, protocol, ec);
  }

  /// Assign an existing native socket to the socket.
  /*
   * This function opens the socket to hold an existing native socket.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @param native_socket A native socket.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void assign(const protocol_type& protocol, const native_type& native_socket)
  {
    asio::error_code ec;
    this->service.assign(this->implementation, protocol, native_socket, ec);
    asio::detail::throw_error(ec);
  }

  /// Assign an existing native socket to the socket.
  /*
   * This function opens the socket to hold an existing native socket.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @param native_socket A native socket.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  asio::error_code assign(const protocol_type& protocol,
      const native_type& native_socket, asio::error_code& ec)
  {
    return this->service.assign(this->implementation,
        protocol, native_socket, ec);
  }

  /// Close the socket.
  /**
   * This function is used to close the socket. Any asynchronous send, receive
   * or connect operations will be cancelled immediately, and will complete
   * with the asio::error::operation_aborted error.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void close()
  {
    asio::error_code ec;
    this->service.close(this->implementation, ec);
    asio::detail::throw_error(ec);
  }

  /// Close the socket.
  /**
   * This function is used to close the socket. Any asynchronous send, receive
   * or connect operations will be cancelled immediately, and will complete
   * with the asio::error::operation_aborted error.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::error_code ec;
   * socket.close(ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  asio::error_code close(asio::error_code& ec)
  {
    return this->service.close(this->implementation, ec);
  }

  /// Get the native socket representation.
  /**
   * This function may be used to obtain the underlying representation of the
   * socket. This is intended to allow access to native socket functionality
   * that is not otherwise provided.
   */
  native_type native()
  {
    return this->service.native(this->implementation);
  }

  /// Cancel all asynchronous operations associated with the socket.
  /**
   * This function causes all outstanding asynchronous connect, send and receive
   * operations to finish immediately, and the handlers for cancelled operations
   * will be passed the asio::error::operation_aborted error.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void cancel()
  {
    asio::error_code ec;
    this->service.cancel(this->implementation, ec);
    asio::detail::throw_error(ec);
  }

  /// Cancel all asynchronous operations associated with the socket.
  /**
   * This function causes all outstanding asynchronous connect, send and receive
   * operations to finish immediately, and the handlers for cancelled operations
   * will be passed the asio::error::operation_aborted error.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  asio::error_code cancel(asio::error_code& ec)
  {
    return this->service.cancel(this->implementation, ec);
  }

  /// Bind the socket to the given local endpoint.
  /**
   * This function binds the socket to the specified endpoint on the local
   * machine.
   *
   * @param endpoint An endpoint on the local machine to which the socket will
   * be bound.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * socket.open(asio::ip::tcp::v4());
   * socket.bind(asio::ip::tcp::endpoint(
   *       asio::ip::tcp::v4(), 12345));
   * @endcode
   */
  void bind(const endpoint_type& endpoint)
  {
    asio::error_code ec;
    this->service.bind(this->implementation, endpoint, ec);
    asio::detail::throw_error(ec);
  }

  /// Bind the socket to the given local endpoint.
  /**
   * This function binds the socket to the specified endpoint on the local
   * machine.
   *
   * @param endpoint An endpoint on the local machine to which the socket will
   * be bound.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * socket.open(asio::ip::tcp::v4());
   * asio::error_code ec;
   * socket.bind(asio::ip::tcp::endpoint(
   *       asio::ip::tcp::v4(), 12345), ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  asio::error_code bind(const endpoint_type& endpoint,
      asio::error_code& ec)
  {
    return this->service.bind(this->implementation, endpoint, ec);
  }

  /// Connect the socket to the specified endpoint.
  /**
   * This function is used to connect a socket to the specified remote endpoint.
   * The function call will block until the connection is successfully made or
   * an error occurs.
   *
   * The socket is automatically opened if it is not already open. If the
   * connect fails, and the socket was automatically opened, the socket is
   * returned to the closed state.
   *
   * @param peer_endpoint The remote endpoint to which the socket will be
   * connected.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * asio::ip::tcp::endpoint endpoint(
   *     asio::ip::address::from_string("1.2.3.4"), 12345);
   * socket.connect(endpoint);
   * @endcode
   */
  void connect(const endpoint_type& peer_endpoint)
  {
    asio::error_code ec;
    this->service.connect(this->implementation, peer_endpoint, ec);
    asio::detail::throw_error(ec);
  }

  /// Connect the socket to the specified endpoint.
  /**
   * This function is used to connect a socket to the specified remote endpoint.
   * The function call will block until the connection is successfully made or
   * an error occurs.
   *
   * The socket is automatically opened if it is not already open. If the
   * connect fails, and the socket was automatically opened, the socket is
   * returned to the closed state.
   *
   * @param peer_endpoint The remote endpoint to which the socket will be
   * connected.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * asio::ip::tcp::endpoint endpoint(
   *     asio::ip::address::from_string("1.2.3.4"), 12345);
   * asio::error_code ec;
   * socket.connect(endpoint, ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  asio::error_code connect(const endpoint_type& peer_endpoint,
      asio::error_code& ec)
  {
    return this->service.connect(this->implementation, peer_endpoint, ec);
  }

  /// Start an asynchronous connect.
  /**
   * This function is used to asynchronously connect a socket to the specified
   * remote endpoint. The function call always returns immediately.
   *
   * The socket is automatically opened if it is not already open. If the
   * connect fails, and the socket was automatically opened, the socket is
   * returned to the closed state.
   *
   * @param peer_endpoint The remote endpoint to which the socket will be
   * connected. Copies will be made of the endpoint object as required.
   *
   * @param handler The handler to be called when the connection operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error // Result of operation
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @par Example:
   * @code
   * void connect_handler(const asio::error_code& error)
   * {
   *   if (!error)
   *   {
   *     // Connect succeeded.
   *   }
   * }
   *
   * ...
   *
   * asio::ip::tcp::socket socket(io_service);
   * asio::ip::tcp::endpoint endpoint(
   *     asio::ip::address::from_string("1.2.3.4"), 12345);
   * socket.async_connect(endpoint, connect_handler);
   * @endcode
   */
  template <typename Handler>
  void async_connect(const endpoint_type& peer_endpoint, Handler handler)
  {
    this->service.async_connect(this->implementation, peer_endpoint, handler);
  }

  /// Set an option on the socket.
  /**
   * This function is used to set an option on the socket.
   *
   * @param option The new option value to be set on the socket.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @sa Socket_Option @n
   * asio::socket_base::broadcast @n
   * asio::socket_base::do_not_route @n
   * asio::socket_base::keep_alive @n
   * asio::socket_base::linger @n
   * asio::socket_base::receive_buffer_size @n
   * asio::socket_base::receive_low_watermark @n
   * asio::socket_base::reuse_address @n
   * asio::socket_base::send_buffer_size @n
   * asio::socket_base::send_low_watermark @n
   * asio::ip::multicast::join_group @n
   * asio::ip::multicast::leave_group @n
   * asio::ip::multicast::enable_loopback @n
   * asio::ip::multicast::outbound_interface @n
   * asio::ip::multicast::hops @n
   * asio::ip::tcp::no_delay
   *
   * @par Example:
   * Setting the IPPROTO_TCP/TCP_NODELAY option:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::ip::tcp::no_delay option(true);
   * socket.set_option(option);
   * @endcode
   */
  template <typename Socket_Option>
  void set_option(const Socket_Option& option)
  {
    asio::error_code ec;
    this->service.set_option(this->implementation, option, ec);
    asio::detail::throw_error(ec);
  }

  /// Set an option on the socket.
  /**
   * This function is used to set an option on the socket.
   *
   * @param option The new option value to be set on the socket.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @sa Socket_Option @n
   * asio::socket_base::broadcast @n
   * asio::socket_base::do_not_route @n
   * asio::socket_base::keep_alive @n
   * asio::socket_base::linger @n
   * asio::socket_base::receive_buffer_size @n
   * asio::socket_base::receive_low_watermark @n
   * asio::socket_base::reuse_address @n
   * asio::socket_base::send_buffer_size @n
   * asio::socket_base::send_low_watermark @n
   * asio::ip::multicast::join_group @n
   * asio::ip::multicast::leave_group @n
   * asio::ip::multicast::enable_loopback @n
   * asio::ip::multicast::outbound_interface @n
   * asio::ip::multicast::hops @n
   * asio::ip::tcp::no_delay
   *
   * @par Example:
   * Setting the IPPROTO_TCP/TCP_NODELAY option:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::ip::tcp::no_delay option(true);
   * asio::error_code ec;
   * socket.set_option(option, ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Socket_Option>
  asio::error_code set_option(const Socket_Option& option,
      asio::error_code& ec)
  {
    return this->service.set_option(this->implementation, option, ec);
  }

  /// Get an option from the socket.
  /**
   * This function is used to get the current value of an option on the socket.
   *
   * @param option The option value to be obtained from the socket.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @sa Socket_Option @n
   * asio::socket_base::broadcast @n
   * asio::socket_base::do_not_route @n
   * asio::socket_base::keep_alive @n
   * asio::socket_base::linger @n
   * asio::socket_base::receive_buffer_size @n
   * asio::socket_base::receive_low_watermark @n
   * asio::socket_base::reuse_address @n
   * asio::socket_base::send_buffer_size @n
   * asio::socket_base::send_low_watermark @n
   * asio::ip::multicast::join_group @n
   * asio::ip::multicast::leave_group @n
   * asio::ip::multicast::enable_loopback @n
   * asio::ip::multicast::outbound_interface @n
   * asio::ip::multicast::hops @n
   * asio::ip::tcp::no_delay
   *
   * @par Example:
   * Getting the value of the SOL_SOCKET/SO_KEEPALIVE option:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::ip::tcp::socket::keep_alive option;
   * socket.get_option(option);
   * bool is_set = option.get();
   * @endcode
   */
  template <typename Socket_Option>
  void get_option(Socket_Option& option) const
  {
    asio::error_code ec;
    this->service.get_option(this->implementation, option, ec);
    asio::detail::throw_error(ec);
  }

  /// Get an option from the socket.
  /**
   * This function is used to get the current value of an option on the socket.
   *
   * @param option The option value to be obtained from the socket.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @sa Socket_Option @n
   * asio::socket_base::broadcast @n
   * asio::socket_base::do_not_route @n
   * asio::socket_base::keep_alive @n
   * asio::socket_base::linger @n
   * asio::socket_base::receive_buffer_size @n
   * asio::socket_base::receive_low_watermark @n
   * asio::socket_base::reuse_address @n
   * asio::socket_base::send_buffer_size @n
   * asio::socket_base::send_low_watermark @n
   * asio::ip::multicast::join_group @n
   * asio::ip::multicast::leave_group @n
   * asio::ip::multicast::enable_loopback @n
   * asio::ip::multicast::outbound_interface @n
   * asio::ip::multicast::hops @n
   * asio::ip::tcp::no_delay
   *
   * @par Example:
   * Getting the value of the SOL_SOCKET/SO_KEEPALIVE option:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::ip::tcp::socket::keep_alive option;
   * asio::error_code ec;
   * socket.get_option(option, ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * bool is_set = option.get();
   * @endcode
   */
  template <typename Socket_Option>
  asio::error_code get_option(Socket_Option& option,
      asio::error_code& ec) const
  {
    return this->service.get_option(this->implementation, option, ec);
  }

  /// Perform an IO control command on the socket.
  /**
   * This function is used to execute an IO control command on the socket.
   *
   * @param command The IO control command to be performed on the socket.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @sa IO_Control_Command @n
   * asio::socket_base::bytes_readable @n
   * asio::socket_base::non_blocking_io
   *
   * @par Example:
   * Getting the number of bytes ready to read:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::ip::tcp::socket::bytes_readable command;
   * socket.io_control(command);
   * std::size_t bytes_readable = command.get();
   * @endcode
   */
  template <typename IO_Control_Command>
  void io_control(IO_Control_Command& command)
  {
    asio::error_code ec;
    this->service.io_control(this->implementation, command, ec);
    asio::detail::throw_error(ec);
  }

  /// Perform an IO control command on the socket.
  /**
   * This function is used to execute an IO control command on the socket.
   *
   * @param command The IO control command to be performed on the socket.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @sa IO_Control_Command @n
   * asio::socket_base::bytes_readable @n
   * asio::socket_base::non_blocking_io
   *
   * @par Example:
   * Getting the number of bytes ready to read:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::ip::tcp::socket::bytes_readable command;
   * asio::error_code ec;
   * socket.io_control(command, ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * std::size_t bytes_readable = command.get();
   * @endcode
   */
  template <typename IO_Control_Command>
  asio::error_code io_control(IO_Control_Command& command,
      asio::error_code& ec)
  {
    return this->service.io_control(this->implementation, command, ec);
  }

  /// Get the local endpoint of the socket.
  /**
   * This function is used to obtain the locally bound endpoint of the socket.
   *
   * @returns An object that represents the local endpoint of the socket.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::ip::tcp::endpoint endpoint = socket.local_endpoint();
   * @endcode
   */
  endpoint_type local_endpoint() const
  {
    asio::error_code ec;
    endpoint_type ep = this->service.local_endpoint(this->implementation, ec);
    asio::detail::throw_error(ec);
    return ep;
  }

  /// Get the local endpoint of the socket.
  /**
   * This function is used to obtain the locally bound endpoint of the socket.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns An object that represents the local endpoint of the socket.
   * Returns a default-constructed endpoint object if an error occurred.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::error_code ec;
   * asio::ip::tcp::endpoint endpoint = socket.local_endpoint(ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  endpoint_type local_endpoint(asio::error_code& ec) const
  {
    return this->service.local_endpoint(this->implementation, ec);
  }

  /// Get the remote endpoint of the socket.
  /**
   * This function is used to obtain the remote endpoint of the socket.
   *
   * @returns An object that represents the remote endpoint of the socket.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::ip::tcp::endpoint endpoint = socket.remote_endpoint();
   * @endcode
   */
  endpoint_type remote_endpoint() const
  {
    asio::error_code ec;
    endpoint_type ep = this->service.remote_endpoint(this->implementation, ec);
    asio::detail::throw_error(ec);
    return ep;
  }

  /// Get the remote endpoint of the socket.
  /**
   * This function is used to obtain the remote endpoint of the socket.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns An object that represents the remote endpoint of the socket.
   * Returns a default-constructed endpoint object if an error occurred.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::error_code ec;
   * asio::ip::tcp::endpoint endpoint = socket.remote_endpoint(ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  endpoint_type remote_endpoint(asio::error_code& ec) const
  {
    return this->service.remote_endpoint(this->implementation, ec);
  }

  /// Disable sends or receives on the socket.
  /**
   * This function is used to disable send operations, receive operations, or
   * both.
   *
   * @param what Determines what types of operation will no longer be allowed.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @par Example:
   * Shutting down the send side of the socket:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * socket.shutdown(asio::ip::tcp::socket::shutdown_send);
   * @endcode
   */
  void shutdown(shutdown_type what)
  {
    asio::error_code ec;
    this->service.shutdown(this->implementation, what, ec);
    asio::detail::throw_error(ec);
  }

  /// Disable sends or receives on the socket.
  /**
   * This function is used to disable send operations, receive operations, or
   * both.
   *
   * @param what Determines what types of operation will no longer be allowed.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @par Example:
   * Shutting down the send side of the socket:
   * @code
   * asio::ip::tcp::socket socket(io_service);
   * ...
   * asio::error_code ec;
   * socket.shutdown(asio::ip::tcp::socket::shutdown_send, ec);
   * if (ec)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  asio::error_code shutdown(shutdown_type what,
      asio::error_code& ec)
  {
    return this->service.shutdown(this->implementation, what, ec);
  }

protected:
  /// Protected destructor to prevent deletion through this type.
  ~basic_socket()
  {
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_SOCKET_HPP
