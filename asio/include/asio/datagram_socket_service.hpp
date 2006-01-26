//
// datagram_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DATAGRAM_SOCKET_SERVICE_HPP
#define ASIO_DATAGRAM_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_io_service.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/kqueue_reactor.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/reactive_socket_service.hpp"
#include "asio/detail/win_iocp_socket_service.hpp"

namespace asio {

/// Default service implementation for a datagram socket.
template <typename Protocol, typename Allocator>
class datagram_socket_service
  : private noncopyable
{
public:
  /// The io_service type.
  typedef basic_io_service<Allocator> io_service_type;

  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

private:
  // The type of the platform-specific implementation.
#if defined(ASIO_HAS_IOCP)
  typedef detail::win_iocp_socket_service<Allocator> service_impl_type;
#elif defined(ASIO_HAS_EPOLL)
  typedef detail::reactive_socket_service<io_service_type,
      detail::epoll_reactor<false, Allocator> > service_impl_type;
#elif defined(ASIO_HAS_KQUEUE)
  typedef detail::reactive_socket_service<io_service_type,
      detail::kqueue_reactor<false, Allocator> > service_impl_type;
#else
  typedef detail::reactive_socket_service<io_service_type,
      detail::select_reactor<false, Allocator> > service_impl_type;
#endif

public:
  /// The type of a datagram socket.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef typename service_impl_type::impl_type implementation_type;
#endif

  /// The native socket type.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_type;
#else
  typedef typename service_impl_type::native_type native_type;
#endif

  /// Construct a new datagram socket service for the specified io_service.
  explicit datagram_socket_service(io_service_type& io_service)
    : service_impl_(io_service.get_service(
          service_factory<service_impl_type>()))
  {
  }

  /// Get the io_service associated with the service.
  io_service_type& io_service()
  {
    return service_impl_.io_service();
  }

  /// Construct a new datagram socket implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a datagram socket implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  // Open a new datagram socket implementation.
  template <typename Error_Handler>
  void open(implementation_type& impl, const protocol_type& protocol,
      Error_Handler error_handler)
  {
    if (protocol.type() == SOCK_DGRAM)
      service_impl_.open(impl, protocol, error_handler);
    else
      error_handler(asio::error(asio::error::invalid_argument));
  }

  /// Open a datagram socket from an existing native socket.
  template <typename Error_Handler>
  void open(implementation_type& impl, const native_type& native_socket,
      Error_Handler error_handler)
  {
    service_impl_.open(impl, native_socket, error_handler);
  }

  /// Close a datagram socket implementation.
  template <typename Error_Handler>
  void close(implementation_type& impl, Error_Handler error_handler)
  {
    service_impl_.close(impl, error_handler);
  }

  /// Get the native socket implementation.
  native_type native(implementation_type& impl)
  {
    return service_impl_.native(impl);
  }

  // Bind the datagram socket to the specified local endpoint.
  template <typename Error_Handler>
  void bind(implementation_type& impl, const endpoint_type& endpoint,
      Error_Handler error_handler)
  {
    service_impl_.bind(impl, endpoint, error_handler);
  }

  /// Connect the datagram socket to the specified endpoint.
  template <typename Error_Handler>
  void connect(implementation_type& impl, const endpoint_type& peer_endpoint,
      Error_Handler error_handler)
  {
    service_impl_.connect(impl, peer_endpoint, error_handler);
  }

  /// Start an asynchronous connect.
  template <typename Handler>
  void async_connect(implementation_type& impl,
      const endpoint_type& peer_endpoint, Handler handler)
  {
    service_impl_.async_connect(impl, peer_endpoint, handler);
  }

  /// Set a socket option.
  template <typename Option, typename Error_Handler>
  void set_option(implementation_type& impl, const Option& option,
      Error_Handler error_handler)
  {
    service_impl_.set_option(impl, option, error_handler);
  }

  /// Get a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(const implementation_type& impl, Option& option,
      Error_Handler error_handler) const
  {
    service_impl_.get_option(impl, option, error_handler);
  }

  /// Perform an IO control command on the socket.
  template <typename IO_Control_Command, typename Error_Handler>
  void io_control(implementation_type& impl, IO_Control_Command& command,
      Error_Handler error_handler)
  {
    service_impl_.io_control(impl, command, error_handler);
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

  /// Get the remote endpoint.
  template <typename Error_Handler>
  endpoint_type remote_endpoint(const implementation_type& impl,
      Error_Handler error_handler) const
  {
    endpoint_type endpoint;
    service_impl_.get_remote_endpoint(impl, endpoint, error_handler);
    return endpoint;
  }

  /// Disable sends or receives on the socket.
  template <typename Error_Handler>
  void shutdown(implementation_type& impl, socket_base::shutdown_type what,
      Error_Handler error_handler)
  {
    service_impl_.shutdown(impl, what, error_handler);
  }

  /// Send the given data to the peer.
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t send(implementation_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    return service_impl_.send(impl, buffers, flags, error_handler);
  }

  /// Start an asynchronous send.
  template <typename Const_Buffers, typename Handler>
  void async_send(implementation_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    service_impl_.async_send(impl, buffers, flags, handler);
  }

  /// Send a datagram to the specified endpoint.
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t send_to(implementation_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, const endpoint_type& destination,
      Error_Handler error_handler)
  {
    return service_impl_.send_to(impl, buffers, flags, destination,
        error_handler);
  }

  /// Start an asynchronous send.
  template <typename Const_Buffers, typename Handler>
  void async_send_to(implementation_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, const endpoint_type& destination,
      Handler handler)
  {
    service_impl_.async_send_to(impl, buffers, flags, destination, handler);
  }

  /// Receive some data from the peer.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t receive(implementation_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    return service_impl_.receive(impl, buffers, flags, error_handler);
  }

  /// Start an asynchronous receive.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(implementation_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    service_impl_.async_receive(impl, buffers, flags, handler);
  }

  /// Receive a datagram with the endpoint of the sender.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t receive_from(implementation_type& impl,
      const Mutable_Buffers& buffers, socket_base::message_flags flags,
      endpoint_type& sender_endpoint, Error_Handler error_handler)
  {
    return service_impl_.receive_from(impl, buffers, flags, sender_endpoint,
        error_handler);
  }

  /// Start an asynchronous receive that will get the endpoint of the sender.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive_from(implementation_type& impl,
      const Mutable_Buffers& buffers, socket_base::message_flags flags,
      endpoint_type& sender_endpoint, Handler handler)
  {
    service_impl_.async_receive_from(impl, buffers, flags, sender_endpoint,
        handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DATAGRAM_SOCKET_SERVICE_HPP
