//
// socket_acceptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SOCKET_ACCEPTOR_SERVICE_HPP
#define ASIO_SOCKET_ACCEPTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_socket.hpp"
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/kqueue_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/service_base.hpp"
#include "asio/detail/reactive_socket_service.hpp"
#include "asio/detail/win_iocp_socket_service.hpp"

namespace asio {

/// Default service implementation for a socket acceptor.
template <typename Protocol>
class socket_acceptor_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_service::service
#else
  : public asio::detail::service_base<socket_acceptor_service<Protocol> >
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_service::id id;
#endif

  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename protocol_type::endpoint endpoint_type;

private:
  // The type of the platform-specific implementation.
#if defined(ASIO_HAS_IOCP)
  typedef detail::win_iocp_socket_service<Protocol> service_impl_type;
#elif defined(ASIO_HAS_EPOLL)
  typedef detail::reactive_socket_service<
      Protocol, detail::epoll_reactor<false> > service_impl_type;
#elif defined(ASIO_HAS_KQUEUE)
  typedef detail::reactive_socket_service<
      Protocol, detail::kqueue_reactor<false> > service_impl_type;
#elif defined(ASIO_HAS_DEV_POLL)
  typedef detail::reactive_socket_service<
      Protocol, detail::dev_poll_reactor<false> > service_impl_type;
#else
  typedef detail::reactive_socket_service<
      Protocol, detail::select_reactor<false> > service_impl_type;
#endif

public:
  /// The native type of the socket acceptor.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef typename service_impl_type::implementation_type implementation_type;
#endif

  /// The native acceptor type.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_type;
#else
  typedef typename service_impl_type::native_type native_type;
#endif

  /// Construct a new socket acceptor service for the specified io_service.
  explicit socket_acceptor_service(asio::io_service& io_service)
    : asio::detail::service_base<
        socket_acceptor_service<Protocol> >(io_service),
      service_impl_(asio::use_service<service_impl_type>(io_service))
  {
  }

  /// Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
  }

  /// Construct a new socket acceptor implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a socket acceptor implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Open a new socket acceptor implementation.
  asio::error_code open(implementation_type& impl,
      const protocol_type& protocol, asio::error_code& ec)
  {
    return service_impl_.open(impl, protocol, ec);
  }

  /// Assign an existing native acceptor to a socket acceptor.
  asio::error_code assign(implementation_type& impl,
      const protocol_type& protocol, const native_type& native_acceptor,
      asio::error_code& ec)
  {
    return service_impl_.assign(impl, protocol, native_acceptor, ec);
  }

  /// Determine whether the acceptor is open.
  bool is_open(const implementation_type& impl) const
  {
    return service_impl_.is_open(impl);
  }

  /// Cancel all asynchronous operations associated with the acceptor.
  asio::error_code cancel(implementation_type& impl,
      asio::error_code& ec)
  {
    return service_impl_.cancel(impl, ec);
  }

  /// Bind the socket acceptor to the specified local endpoint.
  asio::error_code bind(implementation_type& impl,
      const endpoint_type& endpoint, asio::error_code& ec)
  {
    return service_impl_.bind(impl, endpoint, ec);
  }

  /// Place the socket acceptor into the state where it will listen for new
  /// connections.
  asio::error_code listen(implementation_type& impl, int backlog,
      asio::error_code& ec)
  {
    return service_impl_.listen(impl, backlog, ec);
  }

  /// Close a socket acceptor implementation.
  asio::error_code close(implementation_type& impl,
      asio::error_code& ec)
  {
    return service_impl_.close(impl, ec);
  }

  /// Get the native acceptor implementation.
  native_type native(implementation_type& impl)
  {
    return service_impl_.native(impl);
  }

  /// Set a socket option.
  template <typename SettableSocketOption>
  asio::error_code set_option(implementation_type& impl,
      const SettableSocketOption& option, asio::error_code& ec)
  {
    return service_impl_.set_option(impl, option, ec);
  }

  /// Get a socket option.
  template <typename GettableSocketOption>
  asio::error_code get_option(const implementation_type& impl,
      GettableSocketOption& option, asio::error_code& ec) const
  {
    return service_impl_.get_option(impl, option, ec);
  }

  /// Perform an IO control command on the socket.
  template <typename IoControlCommand>
  asio::error_code io_control(implementation_type& impl,
      IoControlCommand& command, asio::error_code& ec)
  {
    return service_impl_.io_control(impl, command, ec);
  }

  /// Get the local endpoint.
  endpoint_type local_endpoint(const implementation_type& impl,
      asio::error_code& ec) const
  {
    return service_impl_.local_endpoint(impl, ec);
  }

  /// Accept a new connection.
  template <typename SocketService>
  asio::error_code accept(implementation_type& impl,
      basic_socket<protocol_type, SocketService>& peer,
      endpoint_type* peer_endpoint, asio::error_code& ec)
  {
    return service_impl_.accept(impl, peer, peer_endpoint, ec);
  }

  /// Start an asynchronous accept.
  template <typename SocketService, typename AcceptHandler>
  void async_accept(implementation_type& impl,
      basic_socket<protocol_type, SocketService>& peer,
      endpoint_type* peer_endpoint, AcceptHandler handler)
  {
    service_impl_.async_accept(impl, peer, peer_endpoint, handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_ACCEPTOR_SERVICE_HPP
