//
// socket_acceptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
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

#include "asio/detail/push_options.hpp"
#include <memory>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/basic_stream_socket.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_socket_acceptor_service.hpp"

namespace asio {

/// Default service implementation for a socket acceptor.
template <typename Allocator = std::allocator<void> >
class socket_acceptor_service
  : private boost::noncopyable
{
public:
  /// The demuxer type.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

private:
  // The type of the platform-specific implementation.
#if defined(_WIN32)
  typedef detail::reactive_socket_acceptor_service<
    demuxer_type, detail::select_reactor<true> > service_impl_type;
#elif defined(ASIO_HAS_EPOLL_REACTOR)
  typedef detail::reactive_socket_acceptor_service<
    demuxer_type, detail::epoll_reactor<false> > service_impl_type;
#else
  typedef detail::reactive_socket_acceptor_service<
    demuxer_type, detail::select_reactor<false> > service_impl_type;
#endif

public:
  /// The native type of the socket acceptor.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined impl_type;
#else
  typedef typename service_impl_type::impl_type impl_type;
#endif

  /// Construct a new socket acceptor service for the specified demuxer.
  socket_acceptor_service(demuxer_type& demuxer)
    : service_impl_(demuxer.get_service(service_factory<service_impl_type>()))
  {
  }

  /// Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return service_impl_.demuxer();
  }

  /// Return a null socket acceptor implementation.
  impl_type null() const
  {
    return service_impl_.null();
  }

  /// Open a new socket acceptor implementation.
  template <typename Protocol, typename Error_Handler>
  void open(impl_type& impl, const Protocol& protocol,
      Error_Handler error_handler)
  {
    service_impl_.open(impl, protocol, error_handler);
  }

  /// Bind the socket acceptor to the specified local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void bind(impl_type& impl, const Endpoint& endpoint,
      Error_Handler error_handler)
  {
    service_impl_.bind(impl, endpoint, error_handler);
  }

  /// Place the socket acceptor into the state where it will listen for new
  /// connections.
  template <typename Error_Handler>
  void listen(impl_type& impl, int backlog, Error_Handler error_handler)
  {
    service_impl_.listen(impl, backlog, error_handler);
  }

  /// Close a socket acceptor implementation.
  void close(impl_type& impl)
  {
    service_impl_.close(impl);
  }

  /// Set a socket option.
  template <typename Option, typename Error_Handler>
  void set_option(impl_type& impl, const Option& option,
      Error_Handler error_handler)
  {
    service_impl_.set_option(impl, option, error_handler);
  }

  /// Set a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(impl_type& impl, Option& option, Error_Handler error_handler)
  {
    service_impl_.get_option(impl, option, error_handler);
  }

  /// Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler)
  {
    service_impl_.get_local_endpoint(impl, endpoint, error_handler);
  }

  /// Accept a new connection.
  template <typename Stream_Socket_Service, typename Error_Handler>
  void accept(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      Error_Handler error_handler)
  {
    service_impl_.accept(impl, peer, error_handler);
  }

  /// Accept a new connection.
  template <typename Stream_Socket_Service, typename Endpoint,
      typename Error_Handler>
  void accept_endpoint(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      Endpoint& peer_endpoint, Error_Handler error_handler)
  {
    service_impl_.accept_endpoint(impl, peer, peer_endpoint, error_handler);
  }

  /// Start an asynchronous accept.
  template <typename Stream_Socket_Service, typename Handler>
  void async_accept(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer, Handler handler)
  {
    service_impl_.async_accept(impl, peer, handler);
  }

  /// Start an asynchronous accept.
  template <typename Stream_Socket_Service, typename Endpoint,
      typename Handler>
  void async_accept_endpoint(impl_type& impl,
      basic_stream_socket<Stream_Socket_Service>& peer,
      Endpoint& peer_endpoint, Handler handler)
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
