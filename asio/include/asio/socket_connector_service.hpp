//
// socket_connector_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SOCKET_CONNECTOR_SERVICE_HPP
#define ASIO_SOCKET_CONNECTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <memory>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/basic_stream_socket.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_socket_connector_service.hpp"

namespace asio {

/// Default service implementation for a socket connector.
template <typename Allocator = std::allocator<void> >
class socket_connector_service
  : private boost::noncopyable
{
public:
  /// The demuxer type.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

private:
  // The type of the platform-specific implementation.
#if defined(_WIN32)
  typedef detail::reactive_socket_connector_service<
    demuxer_type, detail::select_reactor<true> > service_impl_type;
#elif defined(ASIO_HAS_EPOLL_REACTOR)
  typedef detail::reactive_socket_connector_service<
    demuxer_type, detail::epoll_reactor<false> > service_impl_type;
#else
  typedef detail::reactive_socket_connector_service<
    demuxer_type, detail::select_reactor<false> > service_impl_type;
#endif

public:
  /// The native type of the socket connector.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined impl_type;
#else
  typedef typename service_impl_type::impl_type impl_type;
#endif

  /// Construct a new socket connector service for the specified demuxer.
  explicit socket_connector_service(demuxer_type& demuxer)
    : service_impl_(demuxer.get_service(service_factory<service_impl_type>()))
  {
  }

  /// Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return service_impl_.demuxer();
  }

  /// Return a null socket connector implementation.
  impl_type null() const
  {
    return service_impl_.null();
  }

  /// Open a new socket connector implementation without specifying a protocol.
  void open(impl_type& impl)
  {
    service_impl_.open(impl);
  }

  /// Open a new socket connector implementation using the specified protocol.
  template <typename Protocol>
  void open(impl_type& impl, const Protocol& protocol)
  {
    service_impl_.open(impl, protocol);
  }

  /// Close a socket connector implementation.
  void close(impl_type& impl)
  {
    service_impl_.close(impl);
  }

  /// Connect the given socket to the peer at the specified endpoint.
  template <typename Stream_Socket_Service, typename Endpoint,
      typename Error_Handler>
  void connect(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      const Endpoint& peer_endpoint, Error_Handler error_handler)
  {
    service_impl_.connect(impl, peer, peer_endpoint, error_handler);
  }

  /// Start an asynchronous connect.
  template <typename Stream_Socket_Service, typename Endpoint,
      typename Handler>
  void async_connect(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      const Endpoint& peer_endpoint, Handler handler)
  {
    service_impl_.async_connect(impl, peer, peer_endpoint, handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_CONNECTOR_SERVICE_HPP
