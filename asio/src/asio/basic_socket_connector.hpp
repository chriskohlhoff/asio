//
// basic_socket_connector.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_BASIC_SOCKET_CONNECTOR_HPP
#define ASIO_BASIC_SOCKET_CONNECTOR_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/completion_context.hpp"
#include "asio/demuxer.hpp"

namespace asio {

class socket_address;

/// The basic_socket_connector class template is used to connect a socket to a
/// remote endpoint. Most applications will simply use the socket_connector
/// typedef.
template <typename Service>
class basic_socket_connector
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide connect operations.
  typedef Service service_type;

  /// The native implementation type of the socket connector.
  typedef typename service_type::impl_type impl_type;

  /// Constructor a connector. The connector is automatically opened.
  explicit basic_socket_connector(demuxer& d)
    : service_(dynamic_cast<service_type&>(d.get_service(service_type::id))),
      impl_(service_type::invalid_impl)
  {
    service_.create(impl_);
  }

  /// Destructor.
  ~basic_socket_connector()
  {
    service_.destroy(impl_);
  }

  /// Open the connector.
  void open()
  {
    service_.create(impl_);
  }

  /// Close the connector.
  void close()
  {
    service_.destroy(impl_);
  }

  /// Get the underlying implementation in the native type.
  impl_type impl() const
  {
    return impl_;
  }

  /// Connect the given socket to the peer at the specified address. Throws a
  /// socket_error exception on failure.
  template <typename Stream>
  void connect(Stream& peer_socket, const socket_address& peer_address)
  {
    service_.connect(impl_, peer_socket.lowest_layer(), peer_address);
  }

  /// The type of a handler called when the asynchronous connect completes. The
  /// only argument is the error code.
  typedef typename service_type::connect_handler connect_handler;

  /// Start an asynchronous connect. The peer_socket object must be valid until
  /// the connect's completion handler is invoked.
  template <typename Stream>
  void async_connect(Stream& peer_socket,
      const socket_address& peer_address, const connect_handler& handler)
  {
    service_.async_connect(impl_, peer_socket.lowest_layer(), peer_address,
        handler, completion_context::null());
  }

  /// Start an asynchronous connect. The peer_socket object must be valid until
  /// the connect's completion handler is invoked.
  template <typename Stream>
  void async_connect(Stream& peer_socket,
      const socket_address& peer_address, const connect_handler& handler,
      completion_context& context)
  {
    service_.async_connect(impl_, peer_socket.lowest_layer(), peer_address,
        handler, context);
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
