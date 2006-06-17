//
// basic_socket_acceptor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_SOCKET_ACCEPTOR_HPP
#define ASIO_BASIC_SOCKET_ACCEPTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_io_object.hpp"
#include "asio/basic_socket.hpp"
#include "asio/error.hpp"
#include "asio/error_handler.hpp"
#include "asio/socket_acceptor_service.hpp"
#include "asio/socket_base.hpp"

namespace asio {

/// Provides the ability to accept new connections.
/**
 * The basic_socket_acceptor class template is used for accepting new socket
 * connections.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object, Error_Source.
 *
 * @par Example:
 * Opening a socket acceptor with the SO_REUSEADDR option enabled:
 * @code
 * asio::ip::tcp::acceptor acceptor(io_service);
 * asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), port);
 * acceptor.open(endpoint.protocol());
 * acceptor.set_option(asio::ip::tcp::acceptor::reuse_address(true));
 * acceptor.bind(endpoint);
 * acceptor.listen();
 * @endcode
 */
template <typename Protocol,
    typename Service = socket_acceptor_service<Protocol> >
class basic_socket_acceptor
  : public basic_io_object<Service>,
    public socket_base
{
public:
  /// The native representation of an acceptor.
  typedef typename Service::native_type native_type;

  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  /// The type used for reporting errors.
  typedef asio::error error_type;

  /// Construct an acceptor without opening it.
  /**
   * This constructor creates an acceptor without opening it to listen for new
   * connections. The open() function must be called before the acceptor can
   * accept new socket connections.
   *
   * @param io_service The io_service object that the acceptor will use to
   * dispatch handlers for any asynchronous operations performed on the
   * acceptor.
   */
  explicit basic_socket_acceptor(asio::io_service& io_service)
    : basic_io_object<Service>(io_service)
  {
  }

  /// Construct an open acceptor.
  /**
   * This constructor creates an acceptor and automatically opens it.
   *
   * @param io_service The io_service object that the acceptor will use to
   * dispatch handlers for any asynchronous operations performed on the
   * acceptor.
   *
   * @param protocol An object specifying protocol parameters to be used.
   *
   * @throws asio::error Thrown on failure.
   */
  basic_socket_acceptor(asio::io_service& io_service,
      const protocol_type& protocol)
    : basic_io_object<Service>(io_service)
  {
    this->service.open(this->implementation, protocol, throw_error());
  }

  /// Construct an acceptor opened on the given endpoint.
  /**
   * This constructor creates an acceptor and automatically opens it to listen
   * for new connections on the specified endpoint.
   *
   * @param io_service The io_service object that the acceptor will use to
   * dispatch handlers for any asynchronous operations performed on the
   * acceptor.
   *
   * @param endpoint An endpoint on the local machine on which the acceptor
   * will listen for new connections.
   *
   * @param listen_backlog The maximum length of the queue of pending
   * connections. A value of 0 means use the default queue length.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note This constructor is equivalent to the following code:
   * @code
   * basic_socket_acceptor<Protocol> acceptor(io_service);
   * acceptor.open(endpoint.protocol());
   * acceptor.bind(endpoint);
   * acceptor.listen(listen_backlog);
   * @endcode
   */
  basic_socket_acceptor(asio::io_service& io_service,
      const endpoint_type& endpoint, int listen_backlog = 0)
    : basic_io_object<Service>(io_service)
  {
    this->service.open(this->implementation, endpoint.protocol(),
        throw_error());
    this->service.bind(this->implementation, endpoint, throw_error());
    this->service.listen(this->implementation, listen_backlog, throw_error());
  }

  /// Construct a basic_socket_acceptor on an existing native acceptor.
  /**
   * This constructor creates an acceptor object to hold an existing native
   * acceptor.
   *
   * @param io_service The io_service object that the acceptor will use to
   * dispatch handlers for any asynchronous operations performed on the
   * acceptor.
   *
   * @param protocol An object specifying protocol parameters to be used.
   *
   * @param native_acceptor A native acceptor.
   *
   * @throws asio::error Thrown on failure.
   */
  basic_socket_acceptor(asio::io_service& io_service,
      const protocol_type& protocol, const native_type& native_acceptor)
    : basic_io_object<Service>(io_service)
  {
    this->service.assign(this->implementation, protocol, native_acceptor,
        throw_error());
  }

  /// Open the acceptor using the specified protocol.
  /**
   * This function opens the socket acceptor so that it will use the specified
   * protocol.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * acceptor.open(asio::ip::tcp::v4());
   * @endcode
   */
  void open(const protocol_type& protocol = protocol_type())
  {
    this->service.open(this->implementation, protocol, throw_error());
  }

  /// Open the acceptor using the specified protocol.
  /**
   * This function opens the socket acceptor so that it will use the specified
   * protocol.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * asio::error error;
   * acceptor.open(asio::ip::tcp::v4(),
   *     asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Error_Handler>
  void open(const protocol_type& protocol, Error_Handler error_handler)
  {
    this->service.open(this->implementation, protocol, error_handler);
  }

  /// Assigns an existing native acceptor to the acceptor.
  /*
   * This function opens the acceptor to hold an existing native acceptor.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @param native_acceptor A native acceptor.
   *
   * @throws asio::error Thrown on failure.
   */
  void assign(const protocol_type& protocol, const native_type& native_acceptor)
  {
    this->service.assign(this->implementation, protocol, native_acceptor,
        throw_error());
  }

  /// Assigns an existing native acceptor to the acceptor.
  /*
   * This function opens the acceptor to hold an existing native acceptor.
   *
   * @param protocol An object specifying which protocol is to be used.
   *
   * @param native_acceptor A native acceptor.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void assign(const protocol_type& protocol, const native_type& native_acceptor,
      Error_Handler error_handler)
  {
    this->service.assign(this->implementation, protocol, native_acceptor,
        error_handler);
  }

  /// Bind the acceptor to the given local endpoint.
  /**
   * This function binds the socket acceptor to the specified endpoint on the
   * local machine.
   *
   * @param endpoint An endpoint on the local machine to which the socket
   * acceptor will be bound.
   *
   * @throws asio::error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * acceptor.open(asio::ip::tcp::v4());
   * acceptor.bind(asio::ip::tcp::endpoint(12345));
   * @endcode
   */
  void bind(const endpoint_type& endpoint)
  {
    this->service.bind(this->implementation, endpoint, throw_error());
  }

  /// Bind the acceptor to the given local endpoint.
  /**
   * This function binds the socket acceptor to the specified endpoint on the
   * local machine.
   *
   * @param endpoint An endpoint on the local machine to which the socket
   * acceptor will be bound.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * acceptor.open(asio::ip::tcp::v4());
   * asio::error error;
   * acceptor.bind(asio::ip::tcp::endpoint(12345),
   *     asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Error_Handler>
  void bind(const endpoint_type& endpoint, Error_Handler error_handler)
  {
    this->service.bind(this->implementation, endpoint, error_handler);
  }

  /// Place the acceptor into the state where it will listen for new
  /// connections.
  /**
   * This function puts the socket acceptor into the state where it may accept
   * new connections.
   *
   * @param backlog The maximum length of the queue of pending connections. A
   * value of 0 means use the default queue length.
   */
  void listen(int backlog = 0)
  {
    this->service.listen(this->implementation, backlog, throw_error());
  }

  /// Place the acceptor into the state where it will listen for new
  /// connections.
  /**
   * This function puts the socket acceptor into the state where it may accept
   * new connections.
   *
   * @param backlog The maximum length of the queue of pending connections. A
   * value of 0 means use the default queue length.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::error error;
   * acceptor.listen(0, asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Error_Handler>
  void listen(int backlog, Error_Handler error_handler)
  {
    this->service.listen(this->implementation, backlog, error_handler);
  }

  /// Close the acceptor.
  /**
   * This function is used to close the acceptor. Any asynchronous accept
   * operations will be cancelled immediately.
   *
   * A subsequent call to open() is required before the acceptor can again be
   * used to again perform socket accept operations.
   *
   * @throws asio::error Thrown on failure.
   */
  void close()
  {
    this->service.close(this->implementation, throw_error());
  }

  /// Close the acceptor.
  /**
   * This function is used to close the acceptor. Any asynchronous accept
   * operations will be cancelled immediately.
   *
   * A subsequent call to open() is required before the acceptor can again be
   * used to again perform socket accept operations.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::error error;
   * acceptor.close(asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Error_Handler>
  void close(Error_Handler error_handler)
  {
    this->service.close(this->implementation, error_handler);
  }

  /// Get the native acceptor representation.
  /**
   * This function may be used to obtain the underlying representation of the
   * acceptor. This is intended to allow access to native acceptor functionality
   * that is not otherwise provided.
   */
  native_type native()
  {
    return this->service.native(this->implementation);
  }

  /// Set an option on the acceptor.
  /**
   * This function is used to set an option on the acceptor.
   *
   * @param option The new option value to be set on the acceptor.
   *
   * @throws asio::error Thrown on failure.
   *
   * @sa Socket_Option @n
   * asio::socket_base::reuse_address
   * asio::socket_base::enable_connection_aborted
   *
   * @par Example:
   * Setting the SOL_SOCKET/SO_REUSEADDR option:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::acceptor::reuse_address option(true);
   * acceptor.set_option(option);
   * @endcode
   */
  template <typename Option>
  void set_option(const Option& option)
  {
    this->service.set_option(this->implementation, option, throw_error());
  }

  /// Set an option on the acceptor.
  /**
   * This function is used to set an option on the acceptor.
   *
   * @param option The new option value to be set on the acceptor.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @sa Socket_Option @n
   * asio::socket_base::reuse_address
   * asio::socket_base::enable_connection_aborted
   *
   * @par Example:
   * Setting the SOL_SOCKET/SO_REUSEADDR option:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::acceptor::reuse_address option(true);
   * asio::error error;
   * acceptor.set_option(option, asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Option, typename Error_Handler>
  void set_option(const Option& option, Error_Handler error_handler)
  {
    this->service.set_option(this->implementation, option, error_handler);
  }

  /// Get an option from the acceptor.
  /**
   * This function is used to get the current value of an option on the
   * acceptor.
   *
   * @param option The option value to be obtained from the acceptor.
   *
   * @throws asio::error Thrown on failure.
   *
   * @sa Socket_Option @n
   * asio::socket_base::reuse_address
   *
   * @par Example:
   * Getting the value of the SOL_SOCKET/SO_REUSEADDR option:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::acceptor::reuse_address option;
   * acceptor.get_option(option);
   * bool is_set = option.get();
   * @endcode
   */
  template <typename Option>
  void get_option(Option& option)
  {
    this->service.get_option(this->implementation, option, throw_error());
  }

  /// Get an option from the acceptor.
  /**
   * This function is used to get the current value of an option on the
   * acceptor.
   *
   * @param option The option value to be obtained from the acceptor.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @sa Socket_Option @n
   * asio::socket_base::reuse_address
   *
   * @par Example:
   * Getting the value of the SOL_SOCKET/SO_REUSEADDR option:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::acceptor::reuse_address option;
   * asio::error error;
   * acceptor.get_option(option, asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * bool is_set = option.get();
   * @endcode
   */
  template <typename Option, typename Error_Handler>
  void get_option(Option& option, Error_Handler error_handler)
  {
    this->service.get_option(this->implementation, option, error_handler);
  }

  /// Get the local endpoint of the acceptor.
  /**
   * This function is used to obtain the locally bound endpoint of the acceptor.
   *
   * @returns An object that represents the local endpoint of the acceptor.
   *
   * @throws asio::error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::endpoint endpoint = acceptor.local_endpoint();
   * @endcode
   */
  endpoint_type local_endpoint() const
  {
    return this->service.local_endpoint(this->implementation, throw_error());
  }

  /// Get the local endpoint of the acceptor.
  /**
   * This function is used to obtain the locally bound endpoint of the acceptor.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @returns An object that represents the local endpoint of the acceptor.
   * Returns a default-constructed endpoint object if an error occurred and the
   * error handler did not throw an exception.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::error error;
   * asio::ip::tcp::endpoint endpoint
   *   = acceptor.local_endpoint(asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Error_Handler>
  endpoint_type local_endpoint(Error_Handler error_handler) const
  {
    return this->service.local_endpoint(this->implementation, error_handler);
  }

  /// Accept a new connection.
  /**
   * This function is used to accept a new connection from a peer into the
   * given socket. The function call will block until a new connection has been
   * accepted successfully or an error occurs.
   *
   * @param peer The socket into which the new connection will be accepted.
   *
   * @throws asio::error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::socket socket(io_service);
   * acceptor.accept(socket);
   * @endcode
   */
  template <typename Socket_Service>
  void accept(basic_socket<protocol_type, Socket_Service>& peer)
  {
    this->service.accept(this->implementation, peer, throw_error());
  }

  /// Accept a new connection.
  /**
   * This function is used to accept a new connection from a peer into the
   * given socket. The function call will block until a new connection has been
   * accepted successfully or an error occurs.
   *
   * @param peer The socket into which the new connection will be accepted.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::soocket socket(io_service);
   * asio::error error;
   * acceptor.accept(socket, asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Socket_Service, typename Error_Handler>
  void accept(basic_socket<protocol_type, Socket_Service>& peer,
      Error_Handler error_handler)
  {
    this->service.accept(this->implementation, peer, error_handler);
  }

  /// Start an asynchronous accept.
  /**
   * This function is used to asynchronously accept a new connection into a
   * socket. The function call always returns immediately.
   *
   * @param peer The socket into which the new connection will be accepted.
   * Ownership of the peer object is retained by the caller, which must
   * guarantee that it is valid until the handler is called.
   *
   * @param handler The handler to be called when the accept operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @par Example:
   * @code
   * void accept_handler(const asio::error& error)
   * {
   *   if (!error)
   *   {
   *     // Accept succeeded.
   *   }
   * }
   *
   * ...
   *
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::socket socket(io_service);
   * acceptor.async_accept(socket, accept_handler);
   * @endcode
   */
  template <typename Socket_Service, typename Handler>
  void async_accept(basic_socket<protocol_type, Socket_Service>& peer,
      Handler handler)
  {
    this->service.async_accept(this->implementation, peer, handler);
  }

  /// Accept a new connection and obtain the endpoint of the peer
  /**
   * This function is used to accept a new connection from a peer into the
   * given socket, and additionally provide the endpoint of the remote peer.
   * The function call will block until a new connection has been accepted
   * successfully or an error occurs.
   *
   * @param peer The socket into which the new connection will be accepted.
   *
   * @param peer_endpoint An endpoint object which will receive the endpoint of
   * the remote peer.
   *
   * @throws asio::error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::socket socket(io_service);
   * asio::ip::tcp::endpoint endpoint;
   * acceptor.accept_endpoint(socket, endpoint);
   * @endcode
   */
  template <typename Socket_Service>
  void accept_endpoint(basic_socket<protocol_type, Socket_Service>& peer,
      endpoint_type& peer_endpoint)
  {
    this->service.accept_endpoint(this->implementation, peer, peer_endpoint,
        throw_error());
  }

  /// Accept a new connection and obtain the endpoint of the peer
  /**
   * This function is used to accept a new connection from a peer into the
   * given socket, and additionally provide the endpoint of the remote peer.
   * The function call will block until a new connection has been accepted
   * successfully or an error occurs.
   *
   * @param peer The socket into which the new connection will be accepted.
   *
   * @param peer_endpoint An endpoint object which will receive the endpoint of
   * the remote peer.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @par Example:
   * @code
   * asio::ip::tcp::acceptor acceptor(io_service);
   * ...
   * asio::ip::tcp::socket socket(io_service);
   * asio::ip::tcp::endpoint endpoint;
   * asio::error error;
   * acceptor.accept_endpoint(socket, endpoint,
   *     asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Socket_Service, typename Error_Handler>
  void accept_endpoint(basic_socket<protocol_type, Socket_Service>& peer,
      endpoint_type& peer_endpoint, Error_Handler error_handler)
  {
    this->service.accept_endpoint(this->implementation, peer, peer_endpoint,
        error_handler);
  }

  /// Start an asynchronous accept.
  /**
   * This function is used to asynchronously accept a new connection into a
   * socket, and additionally obtain the endpoint of the remote peer. The
   * function call always returns immediately.
   *
   * @param peer The socket into which the new connection will be accepted.
   * Ownership of the peer object is retained by the caller, which must
   * guarantee that it is valid until the handler is called.
   *
   * @param peer_endpoint An endpoint object into which the endpoint of the
   * remote peer will be written. Ownership of the peer_endpoint object is
   * retained by the caller, which must guarantee that it is valid until the
   * handler is called.
   *
   * @param handler The handler to be called when the accept operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   */
  template <typename Socket_Service, typename Handler>
  void async_accept_endpoint(basic_socket<protocol_type, Socket_Service>& peer,
      endpoint_type& peer_endpoint, Handler handler)
  {
    this->service.async_accept_endpoint(this->implementation, peer,
        peer_endpoint, handler);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_SOCKET_ACCEPTOR_HPP
