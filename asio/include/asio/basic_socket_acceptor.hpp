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
/// connections. Most applications would simply use the socket_acceptor
/// typedef.
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

  /// Constructor an acceptor without opening it. The acceptor needs to be
  /// opened before it can accept new connections.
  explicit basic_socket_acceptor(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
  }

  /// Construct an acceptor opened on the given address.
  template <typename Address>
  basic_socket_acceptor(demuxer_type& d, const Address& addr)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.create(impl_, addr);
  }

  /// Construct an acceptor opened on the given address and with the listen
  /// queue set to the given number of connections.
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
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Open the acceptor using the given address.
  template <typename Address>
  void open(const Address& addr)
  {
    service_.create(impl_, addr);
  }

  /// Open the acceptor using the given address and length of the listen queue.
  template <typename Address>
  void open(const Address& addr, int listen_queue)
  {
    service_.create(impl_, addr, listen_queue);
  }

  /// Close the acceptor.
  void close()
  {
    service_.destroy(impl_);
  }

  /// Get the underlying implementation in the native type.
  impl_type impl()
  {
    return impl_;
  }

  /// Accept a new connection. Throws a socket_error exception on failure.
  template <typename Stream>
  void accept(Stream& peer_socket)
  {
    service_.accept(impl_, peer_socket.lowest_layer());
  }

  /// Accept a new connection. Throws a socket_error exception on failure.
  template <typename Stream, typename Address>
  void accept(Stream& peer_socket, Address& peer_address)
  {
    service_.accept(impl_, peer_socket.lowest_layer(), peer_address);
  }

  /// Start an asynchronous accept. The peer_socket object must be valid until
  /// the accept's completion handler is invoked.
  template <typename Stream, typename Handler>
  void async_accept(Stream& peer_socket, Handler handler)
  {
    service_.async_accept(impl_, peer_socket.lowest_layer(), handler,
        null_completion_context::instance());
  }

  /// Start an asynchronous accept. The peer_socket object must be valid until
  /// the accept's completion handler is invoked.
  template <typename Stream, typename Handler, typename Completion_Context>
  void async_accept(Stream& peer_socket, Handler handler,
      Completion_Context& context)
  {
    service_.async_accept(impl_, peer_socket.lowest_layer(), handler, context);
  }

  /// Start an asynchronous accept. The peer_socket and peer_address objects
  /// must be valid until the accept's completion handler is invoked.
  template <typename Stream, typename Address, typename Handler>
  void async_accept_address(Stream& peer_socket, Address& peer_address,
      Handler handler)
  {
    service_.async_accept_address(impl_, peer_socket.lowest_layer(),
        peer_address, handler, null_completion_context::instance());
  }

  /// Start an asynchronous accept. The peer_socket and peer_address objects
  /// must be valid until the accept's completion handler is invoked.
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
