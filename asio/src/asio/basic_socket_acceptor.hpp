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

#include "asio/completion_context.hpp"
#include "asio/demuxer.hpp"

namespace asio {

class socket_address;

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

  /// Constructor an acceptor without opening it. The acceptor needs to be
  /// opened before it can accept new connections.
  explicit basic_socket_acceptor(demuxer& d)
    : service_(dynamic_cast<service_type&>(d.get_service(service_type::id))),
      impl_(service_type::invalid_impl)
  {
  }

  /// Construct an acceptor opened on the given address.
  basic_socket_acceptor(demuxer& d, const socket_address& addr)
    : service_(dynamic_cast<service_type&>(d.get_service(service_type::id))),
      impl_(service_type::invalid_impl)
  {
    service_.create(impl_, addr);
  }

  /// Construct an acceptor opened on the given address and with the listen
  /// queue set to the given number of connections.
  basic_socket_acceptor(demuxer& d, const socket_address& addr,
      int listen_queue)
    : service_(dynamic_cast<service_type&>(d.get_service(service_type::id))),
      impl_(service_type::invalid_impl)
  {
    service_.create(impl_, addr, listen_queue);
  }

  /// Destructor.
  ~basic_socket_acceptor()
  {
    service_.destroy(impl_);
  }

  /// Open the acceptor using the given address.
  void open(const socket_address& addr)
  {
    service_.create(impl_, addr);
  }

  /// Open the acceptor using the given address and length of the listen queue.
  void open(const socket_address& addr, int listen_queue)
  {
    service_.create(impl_, addr, listen_queue);
  }

  /// Close the acceptor.
  void close()
  {
    service_.destroy(impl_);
  }

  /// Get the underlying implementation in the native type.
  impl_type impl() const
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
  template <typename Stream>
  void accept(Stream& peer_socket, socket_address& peer_address)
  {
    service_.accept(impl_, peer_socket.lowest_layer(), peer_address);
  }

  /// The type of a handler called when the asynchronous accept completes. The
  /// only argument is the error code.
  typedef typename service_type::accept_handler accept_handler;

  /// Start an asynchronous accept. The peer_socket object must be valid until
  /// the accept's completion handler is invoked.
  template <typename Stream>
  void async_accept(Stream& peer_socket, const accept_handler& handler)
  {
    service_.async_accept(impl_, peer_socket.lowest_layer(), handler,
        completion_context::null());
  }

  /// Start an asynchronous accept. The peer_socket object must be valid until
  /// the accept's completion handler is invoked.
  template <typename Stream>
  void async_accept(Stream& peer_socket, const accept_handler& handler,
      completion_context& context)
  {
    service_.async_accept(impl_, peer_socket.lowest_layer(), handler, context);
  }

  /// Start an asynchronous accept. The peer_socket and peer_address objects
  /// must be valid until the accept's completion handler is invoked.
  template <typename Stream>
  void async_accept(Stream& peer_socket, socket_address& peer_address,
      const accept_handler& handler)
  {
    service_.async_accept(impl_, peer_socket.lowest_layer(), peer_address,
        handler, completion_context::null());
  }

  /// Start an asynchronous accept. The peer_socket and peer_address objects
  /// must be valid until the accept's completion handler is invoked.
  template <typename Stream>
  void async_accept(Stream& peer_socket, socket_address& peer_address,
      const accept_handler& handler, completion_context& context)
  {
    service_.async_accept(impl_, peer_socket.lowest_layer(), peer_address,
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

#endif // ASIO_BASIC_SOCKET_ACCEPTOR_HPP
