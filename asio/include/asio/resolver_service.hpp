//
// resolver_service.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_RESOLVER_SERVICE_HPP
#define ASIO_RESOLVER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/io_service.hpp"
#include "asio/detail/resolver_service.hpp"

namespace asio {

/// Default service implementation for a resolver.
template <typename Protocol>
class resolver_service
  : public asio::io_service::service
{
public:
  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  /// The query type.
  typedef typename Protocol::resolver_query query_type;

  /// The iterator type.
  typedef typename Protocol::resolver_iterator iterator_type;

private:
  // The type of the platform-specific implementation.
  typedef detail::resolver_service<Protocol> service_impl_type;

public:
  /// The type of a resolver implementation.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef typename service_impl_type::implementation_type implementation_type;
#endif

  /// Construct a new resolver service for the specified io_service.
  explicit resolver_service(asio::io_service& io_service)
    : asio::io_service::service(io_service),
      service_impl_(asio::use_service<service_impl_type>(io_service))
  {
  }

  /// Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
  }

  /// Construct a new resolver implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a resolver implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Cancel pending asynchronous operations.
  void cancel(implementation_type& impl)
  {
    service_impl_.cancel(impl);
  }

  /// Resolve a query to a list of entries.
  template <typename Error_Handler>
  iterator_type resolve(implementation_type& impl, const query_type& query,
      Error_Handler error_handler)
  {
    return service_impl_.resolve(impl, query, error_handler);
  }

  /// Asynchronously resolve a query to a list of entries.
  template <typename Handler>
  void async_resolve(implementation_type& impl, const query_type& query,
      Handler handler)
  {
    service_impl_.async_resolve(impl, query, handler);
  }

  /// Resolve an endpoint to a list of entries.
  template <typename Error_Handler>
  iterator_type resolve(implementation_type& impl,
      const endpoint_type& endpoint, Error_Handler error_handler)
  {
    return service_impl_.resolve(impl, endpoint, error_handler);
  }

  /// Asynchronously resolve an endpoint to a list of entries.
  template <typename Handler>
  void async_resolve(implementation_type& impl, const endpoint_type& endpoint,
      Handler handler)
  {
    return service_impl_.async_resolve(impl, endpoint, handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_RESOLVER_SERVICE_HPP
