//
// basic_socket_connector.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_SOCKET_CONNECTOR_HPP
#define ASIO_BASIC_SOCKET_CONNECTOR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/default_error_handler.hpp"
#include "asio/service_factory.hpp"

namespace asio {

/// Provides the ability to make new connections.
/**
 * The basic_socket_connector class template is used to connect a socket to a
 * remote endpoint.
 *
 * Most applications will use the asio::socket_connector typedef.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object.
 */
template <typename Service>
class basic_socket_connector
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide connect operations.
  typedef Service service_type;

  /// The native implementation type of the socket connector.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// Constructor.
  /**
   * Constructs the connector and opens it automatically.
   *
   * @param d The demuxer object that the connector will use to dispatch
   * handlers for any asynchronous operations performed on the connector.
   */
  explicit basic_socket_connector(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.open(impl_);
  }

  /// Constructor.
  /**
   * This constructor automatically opens the connector so that it will
   * establish connections using the specified protocol.
   *
   * @param d The demuxer object that the connector will use to dispatch
   * handlers for any asynchronous operations performed on the connector.
   *
   * @param protocol The protocol to be used for all new connections
   * established using the connector.
   */
  template <typename Protocol>
  basic_socket_connector(demuxer_type& d, const Protocol& protocol)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.open(impl_, protocol);
  }

  /// Destructor.
  ~basic_socket_connector()
  {
    service_.close(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the connector
   * uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the demuxer object that the connector will use to
   * dispatch handlers. Ownership is not transferred to the caller.
   */
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Open the connector.
  /**
   * This function is used to open the connector so that it may be used to
   * perform socket connect operations.
   *
   * Since the constructor opens the connector by default, you should only need
   * to call this function if there has been a prior call to close().
   *
   * @throws asio::error Thrown on failure.
   */
  void open()
  {
    service_.open(impl_);
  }

  /// Open the connector to use a specified protocol.
  /**
   * This function is used to open the connector so that it may be used to
   * perform socket connect operations.
   *
   * Since the constructor opens the connector by default, you should only need
   * to call this function if there has been a prior call to close().
   *
   * @param protocol The protocol to be used for all new connections
   * established using the connector.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Protocol>
  void open(const Protocol& protocol)
  {
    service_.open(impl_, protocol);
  }

  /// Close the connector.
  /**
   * This function is used to close the connector. Any asynchronous connect
   * operations will be cancelled immediately.
   *
   * A subsequent call to open() is required before the connector can again be
   * used to again perform socket connect operations.
   */
  void close()
  {
    service_.close(impl_);
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * socket connector. This is intended to allow access to native socket
   * functionality that is not otherwise provided.
   */
  impl_type impl()
  {
    return impl_;
  }

  /// Connect a stream socket to the peer at the specified endpoint.
  /**
   * This function is used to connect a stream socket to the specified remote
   * endpoint. The function call will block until the connection is
   * successfully made or an error occurs.
   *
   * @param peer_socket The stream socket to be connected.
   *
   * @param peer_endpoint The remote endpoint to which the socket will be
   * connected.
   *
   * @throws asio::error Thrown on failure.
   */
  template <typename Stream, typename Endpoint>
  void connect(Stream& peer_socket, const Endpoint& peer_endpoint)
  {
    service_.connect(impl_, peer_socket.lowest_layer(), peer_endpoint,
        default_error_handler());
  }

  /// Connect a stream socket to the peer at the specified endpoint.
  /**
   * This function is used to connect a stream socket to the specified remote
   * endpoint. The function call will block until the connection is successfully
   * made or an error occurs.
   *
   * @param peer_socket The stream socket to be connected.
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
  template <typename Stream, typename Endpoint, typename Error_Handler>
  void connect(Stream& peer_socket, const Endpoint& peer_endpoint,
      Error_Handler error_handler)
  {
    service_.connect(impl_, peer_socket.lowest_layer(), peer_endpoint,
        error_handler);
  }

  /// Start an asynchronous connect.
  /**
   * This function is used to asynchronously connect a stream socket to the
   * specified remote endpoint. The function call always returns immediately.
   *
   * @param peer_socket The stream socket to be connected. Ownership of the
   * peer_socket object is retained by the caller, which must guarantee that it
   * is valid until the handler is called.
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
  template <typename Stream, typename Endpoint, typename Handler>
  void async_connect(Stream& peer_socket, const Endpoint& peer_endpoint,
      Handler handler)
  {
    service_.async_connect(impl_, peer_socket.lowest_layer(), peer_endpoint,
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

#endif // ASIO_BASIC_SOCKET_CONNECTOR_HPP
