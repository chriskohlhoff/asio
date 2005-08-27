//
// stream_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_STREAM_SOCKET_SERVICE_HPP
#define ASIO_STREAM_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <memory>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/demuxer_service.hpp"
#if defined(_WIN32)
# include "asio/detail/win_iocp_stream_socket_service.hpp"
#else
# include "asio/detail/epoll_reactor.hpp"
# include "asio/detail/select_reactor.hpp"
# include "asio/detail/reactive_stream_socket_service.hpp"
#endif

namespace asio {

/// Default service implementation for a stream socket.
template <typename Allocator = std::allocator<void> >
class stream_socket_service
  : private boost::noncopyable
{
public:
  /// The demuxer type.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

private:
  // The type of the platform-specific implementation.
#if defined(_WIN32)
  typedef detail::win_iocp_stream_socket_service<Allocator> service_impl_type;
#elif defined(ASIO_HAS_EPOLL_REACTOR)
  typedef detail::reactive_stream_socket_service<
    demuxer_type, detail::epoll_reactor<false> > service_impl_type;
#else
  typedef detail::reactive_stream_socket_service<
    demuxer_type, detail::select_reactor<false> > service_impl_type;
#endif

public:
  /// The type of a stream socket.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined impl_type;
#else
  typedef typename service_impl_type::impl_type impl_type;
#endif

  /// Construct a new stream socket service for the specified demuxer.
  explicit stream_socket_service(demuxer_type& demuxer)
    : service_impl_(demuxer.get_service(service_factory<service_impl_type>()))
  {
  }

  /// Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return service_impl_.demuxer();
  }

  /// Return a null stream socket implementation.
  impl_type null() const
  {
    return service_impl_.null();
  }

  /// Open a new stream socket implementation.
  void open(impl_type& impl, impl_type new_impl)
  {
    service_impl_.open(impl, new_impl);
  }

  /// Close a stream socket implementation.
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

  /// Get a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(const impl_type& impl, Option& option,
      Error_Handler error_handler) const
  {
    service_impl_.get_option(impl, option, error_handler);
  }

  /// Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(const impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    service_impl_.get_local_endpoint(impl, endpoint, error_handler);
  }

  // Get the remote endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_remote_endpoint(const impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    service_impl_.get_remote_endpoint(impl, endpoint, error_handler);
  }

  /// Disable sends or receives on the socket.
  template <typename Error_Handler>
  void shutdown(impl_type& impl, socket_base::shutdown_type what,
      Error_Handler error_handler)
  {
    service_impl_.shutdown(impl, what, error_handler);
  }

  /// Send the given data to the peer.
  template <typename Error_Handler>
  size_t send(impl_type& impl, const void* data, size_t length,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    return service_impl_.send(impl, data, length, flags, error_handler);
  }

  /// Start an asynchronous send.
  template <typename Handler>
  void async_send(impl_type& impl, const void* data, size_t length,
      socket_base::message_flags flags, Handler handler)
  {
    service_impl_.async_send(impl, data, length, flags, handler);
  }

  /// Receive some data from the peer.
  template <typename Error_Handler>
  size_t receive(impl_type& impl, void* data, size_t max_length,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    return service_impl_.receive(impl, data, max_length, flags, error_handler);
  }

  /// Start an asynchronous receive.
  template <typename Handler>
  void async_receive(impl_type& impl, void* data, size_t max_length,
      socket_base::message_flags flags, Handler handler)
  {
    service_impl_.async_receive(impl, data, max_length, flags, handler);
  }

  /// Determine the amount of data that may be received without blocking.
  template <typename Error_Handler>
  size_t in_avail(impl_type& impl, Error_Handler error_handler)
  {
    return service_impl_.in_avail(impl, error_handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_STREAM_SOCKET_SERVICE_HPP
