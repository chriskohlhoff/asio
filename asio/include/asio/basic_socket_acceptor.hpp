//
// basic_socket_acceptor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_BASIC_SOCKET_ACCEPTOR_HPP
#define ASIO_BASIC_SOCKET_ACCEPTOR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/null_completion_context.hpp"
#include "asio/service_factory.hpp"

namespace asio {

/// The basic_socket_acceptor class template is used for accepting new socket
/// connections. Most applications would use the socket_acceptor typedef.
template <typename Service>
class basic_socket_acceptor
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide accept operations.
  typedef Service service_type;

  /// The native implementation type of the socket acceptor.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// Construct an acceptor without opening it.
  /**
   * This constructor creates an acceptor without opening it to listen for new
   * connections. The open() function must be called before the acceptor can
   * accept new socket connections.
   *
   * @param d The demuxer object that the acceptor will use to deliver
   * completions for any asynchronous operations performed on the acceptor.
   */
  explicit basic_socket_acceptor(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
  }

  /// Construct an acceptor opened on the given address.
  /**
   * This constructor creates an acceptor and automatically opens it to listen
   * for new connections on the specified address.
   *
   * @param d The demuxer object that the acceptor will use to deliver
   * completions for any asynchronous operations performed on the acceptor.
   *
   * @param addr An address on the local machine on which the acceptor will
   * listen for new connections.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Address>
  basic_socket_acceptor(demuxer_type& d, const Address& addr)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.create(impl_, addr);
  }

  /// Construct an acceptor opened on the given address, specifying the
  /// maximum length of the queue of pending connetions.
  /**
   * This constructor creates an acceptor and automatically opens it to listen
   * for new connections on the specified address. The maximum length of the
   * queue of pending connections is also supplied.
   *
   * @param d The demuxer object that the acceptor will use to deliver
   * completions for any asynchronous operations performed on the acceptor.
   *
   * @param addr An address on the local machine on which the acceptor will
   * listen for new connections.
   *
   * @param listen_queue The maximum length of the queue of pending
   * connections.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Address>
  basic_socket_acceptor(demuxer_type& d, const Address& addr, int listen_queue)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.create(impl_, addr, listen_queue);
  }

  /// Destructor.
  ~basic_socket_acceptor()
  {
    service_.destroy(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the acceptor
   * uses to deliver completions for asynchronous operations.
   *
   * @return A reference to the demuxer object that acceptor will use to
   * deliver completion notifications. Ownership is not transferred to the
   * caller.
   */
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Open the acceptor using the given address.
  /**
   * This function opens the acceptor to listen for new connections on the
   * specified address.
   *
   * @param addr An address on the local machine on which the acceptor will
   * listen for new connections.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Address>
  void open(const Address& addr)
  {
    service_.create(impl_, addr);
  }

  /// Open the acceptor using the given address and length of the listen queue.
  /**
   * This function opens the acceptor to listen for new connections on the
   * specified address. The maximum length of the queue of pending connections
   * is also supplied.
   *
   * @param addr An address on the local machine on which the acceptor will
   * listen for new connections.
   *
   * @param listen_queue The maximum length of the queue of pending
   * connections.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Address>
  void open(const Address& addr, int listen_queue)
  {
    service_.create(impl_, addr, listen_queue);
  }

  /// Close the acceptor.
  /**
   * This function is used to close the acceptor. Any asynchronous accept
   * operations will be immediately cancelled.
   *
   * A subsequent call to open() is required before the acceptor can again be
   * used to again perform socket accept operations.
   */
  void close()
  {
    service_.destroy(impl_);
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * socket acceptor. This is intended to allow access to native socket
   * functionality that is not otherwise provided.
   */
  impl_type impl()
  {
    return impl_;
  }

  /// Set an option on the acceptor.
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
    service_.set_option(impl_, option);
  }

  /// Get an option from the acceptor.
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
    service_.get_option(impl_, option);
  }

  /// Accept a new connection.
  /**
   * This function is used to accept a new connection from a peer into the
   * given stream socket. The function call will block until a new connection
   * has been accepted successfully or an error occurs.
   *
   * @param peer_socket The stream socket into which the new connection will be
   * accepted.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Stream>
  void accept(Stream& peer_socket)
  {
    service_.accept(impl_, peer_socket.lowest_layer());
  }

  /// Start an asynchronous accept.
  /**
   * This function is used to asynchronously accept a new connection into a
   * stream socket. The function call always returns immediately.
   *
   * @param peer_socket The stream socket into which the new connection will be
   * accepted. Ownership of the peer_socket object is retained by the caller,
   * which must guarantee that it is valid until the handler is called.
   *
   * @param handler The completion handler to be called when the accept
   * operation completes. Copies will be made of the handler as required. The
   * equivalent function signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   */
  template <typename Stream, typename Handler>
  void async_accept(Stream& peer_socket, Handler handler)
  {
    service_.async_accept(impl_, peer_socket.lowest_layer(), handler,
        null_completion_context::instance());
  }

  /// Start an asynchronous accept.
  /**
   * This function is used to asynchronously accept a new connection into a
   * stream socket. The function call always returns immediately.
   *
   * @param peer_socket The stream socket into which the new connection will be
   * accepted. Ownership of the peer_socket object is retained by the caller,
   * which must guarantee that it is valid until the handler is called.
   *
   * @param handler The completion handler to be called when the accept
   * operation completes. Copies will be made of the handler as required. The
   * equivalent function signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Ownership of the
   * object is retained by the caller, which must guarantee that it is valid
   * until after the handler has been called.
   */
  template <typename Stream, typename Handler, typename Completion_Context>
  void async_accept(Stream& peer_socket, Handler handler,
      Completion_Context& context)
  {
    service_.async_accept(impl_, peer_socket.lowest_layer(), handler, context);
  }

  /// Accept a new connection and obtain the address of the peer
  /**
   * This function is used to accept a new connection from a peer into the
   * given stream socket, and additionally provide the address of the remote
   * peer. The function call will block until a new connection has been
   * accepted successfully or an error occurs.
   *
   * @param peer_socket The stream socket into which the new connection will be
   * accepted.
   *
   * @param peer_address An address object which will receive the network
   * address of the remote peer.
   *
   * @throws socket_error Thrown on failure.
   */
  template <typename Stream, typename Address>
  void accept_address(Stream& peer_socket, Address& peer_address)
  {
    service_.accept(impl_, peer_socket.lowest_layer(), peer_address);
  }

  /// Start an asynchronous accept.
  /**
   * This function is used to asynchronously accept a new connection into a
   * stream socket, and additionally obtain the address of the remote peer. The
   * function call always returns immediately.
   *
   * @param peer_socket The stream socket into which the new connection will be
   * accepted. Ownership of the peer_socket object is retained by the caller,
   * which must guarantee that it is valid until the handler is called.
   *
   * @param peer_address An address object into which the address of the remote
   * peer will be written. Ownership of the peer_address object is retained by
   * the caller, which must guarantee that it is valid until the handler is
   * called.
   *
   * @param handler The completion handler to be called when the accept
   * operation completes. Copies will be made of the handler as required. The
   * equivalent function signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   */
  template <typename Stream, typename Address, typename Handler>
  void async_accept_address(Stream& peer_socket, Address& peer_address,
      Handler handler)
  {
    service_.async_accept_address(impl_, peer_socket.lowest_layer(),
        peer_address, handler, null_completion_context::instance());
  }

  /// Start an asynchronous accept.
  /**
   * This function is used to asynchronously accept a new connection into a
   * stream socket, and additionally obtain the address of the remote peer. The
   * function call always returns immediately.
   *
   * @param peer_socket The stream socket into which the new connection will be
   * accepted. Ownership of the peer_socket object is retained by the caller,
   * which must guarantee that it is valid until the handler is called.
   *
   * @param peer_address An address object into which the address of the remote
   * peer will be written. Ownership of the peer_address object is retained by
   * the caller, which must guarantee that it is valid until the handler is
   * called.
   *
   * @param handler The completion handler to be called when the accept
   * operation completes. Copies will be made of the handler as required. The
   * equivalent function signature of the handler must be:
   * @code void handler(
   *   const asio::socket_error& error // Result of operation
   * ); @endcode
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Ownership of the
   * object is retained by the caller, which must guarantee that it is valid
   * until after the handler has been called.
   */
  template <typename Stream, typename Address, typename Handler,
      typename Completion_Context>
  void async_accept_address(Stream& peer_socket, Address& peer_address,
      Handler handler, Completion_Context& context)
  {
    service_.async_accept_address(impl_, peer_socket.lowest_layer(),
        peer_address, handler, context);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_SOCKET_ACCEPTOR_HPP
