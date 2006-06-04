//
// socket_acceptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include "asio/io_service.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/kqueue_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_socket_service.hpp"
#include "asio/detail/win_iocp_socket_service.hpp"

namespace asio {

/// Default service implementation for a socket acceptor.
template <typename Protocol>
class socket_acceptor_service
  : public asio::io_service::service
{
public:
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
    : asio::io_service::service(io_service),
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
  template <typename Error_Handler>
  void open(implementation_type& impl, const protocol_type& protocol,
      Error_Handler error_handler)
  {
    service_impl_.open(impl, protocol, error_handler);
  }

  /// Assign an existing native acceptor to a socket acceptor.
  template <typename Error_Handler>
  void assign(implementation_type& impl, const protocol_type& protocol,
      const native_type& native_acceptor, Error_Handler error_handler)
  {
    service_impl_.assign(impl, protocol, native_acceptor, error_handler);
  }

  /// Bind the socket acceptor to the specified local endpoint.
  template <typename Error_Handler>
  void bind(implementation_type& impl, const endpoint_type& endpoint,
      Error_Handler error_handler)
  {
    service_impl_.bind(impl, endpoint, error_handler);
  }

  /// Place the socket acceptor into the state where it will listen for new
  /// connections.
  template <typename Error_Handler>
  void listen(implementation_type& impl, int backlog,
      Error_Handler error_handler)
  {
    service_impl_.listen(impl, backlog, error_handler);
  }

  /// Close a socket acceptor implementation.
  template <typename Error_Handler>
  void close(implementation_type& impl, Error_Handler error_handler)
  {
    service_impl_.close(impl, error_handler);
  }

  /// Get the native acceptor implementation.
  native_type native(implementation_type& impl)
  {
    return service_impl_.native(impl);
  }

  /// Set a socket option.
  template <typename Option, typename Error_Handler>
  void set_option(implementation_type& impl, const Option& option,
      Error_Handler error_handler)
  {
    service_impl_.set_option(impl, option, error_handler);
  }

  /// Set a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(implementation_type& impl, Option& option,
      Error_Handler error_handler)
  {
    service_impl_.get_option(impl, option, error_handler);
  }

  /// Get the local endpoint.
  template <typename Error_Handler>
  endpoint_type local_endpoint(const implementation_type& impl,
      Error_Handler error_handler) const
  {
    endpoint_type endpoint;
    service_impl_.get_local_endpoint(impl, endpoint, error_handler);
    return endpoint;
  }

  /// Accept a new connection.
  template <typename Socket_Service, typename Error_Handler>
  void accept(implementation_type& impl,
      basic_socket<protocol_type, Socket_Service>& peer,
      Error_Handler error_handler)
  {
    service_impl_.accept(impl, peer, error_handler);
  }

  /// Accept a new connection.
  template <typename Socket_Service, typename Error_Handler>
  void accept_endpoint(implementation_type& impl,
      basic_socket<protocol_type, Socket_Service>& peer,
      endpoint_type& peer_endpoint, Error_Handler error_handler)
  {
    service_impl_.accept_endpoint(impl, peer, peer_endpoint, error_handler);
  }

  /// Start an asynchronous accept.
  template <typename Socket_Service, typename Handler>
  void async_accept(implementation_type& impl,
      basic_socket<protocol_type, Socket_Service>& peer, Handler handler)
  {
    service_impl_.async_accept(impl, peer, handler);
  }

  /// Start an asynchronous accept.
  template <typename Socket_Service, typename Handler>
  void async_accept_endpoint(implementation_type& impl,
      basic_socket<protocol_type, Socket_Service>& peer,
      endpoint_type& peer_endpoint, Handler handler)
  {
    service_impl_.async_accept_endpoint(impl, peer, peer_endpoint, handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_ACCEPTOR_SERVICE_HPP
