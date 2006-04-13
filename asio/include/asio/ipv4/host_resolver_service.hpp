//
// host_resolver_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_HOST_RESOLVER_SERVICE_HPP
#define ASIO_IPV4_HOST_RESOLVER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/io_service.hpp"
#include "asio/ipv4/host.hpp"
#include "asio/ipv4/detail/host_resolver_service.hpp"

namespace asio {
namespace ipv4 {

/// Default service implementation for a host resolver.
class host_resolver_service
  : public asio::io_service::service
{
private:
  // The type of the platform-specific implementation.
  typedef detail::host_resolver_service service_impl_type;

public:
  /// The type of the host resolver.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef service_impl_type::implementation_type implementation_type;
#endif

  /// Constructor.
  host_resolver_service(asio::io_service& io_service)
    : asio::io_service::service(io_service),
      service_impl_(asio::use_service<service_impl_type>(io_service))
  {
  }

  /// Construct a new host resolver implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a host resolver implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Cancel pending asynchronous operations.
  void cancel(implementation_type& impl)
  {
    service_impl_.cancel(impl);
  }

  /// Get host information for the local machine.
  template <typename Error_Handler>
  void local(implementation_type& impl, host& h, Error_Handler error_handler)
  {
    service_impl_.local(impl, h, error_handler);
  }

  /// Get host information for a specified address.
  template <typename Error_Handler>
  void by_address(implementation_type& impl, host& h, const address& addr,
      Error_Handler error_handler)
  {
    service_impl_.by_address(impl, h, addr, error_handler);
  }

  // Asynchronously get host information for a specified address.
  template <typename Handler>
  void async_by_address(implementation_type& impl, host& h, const address& addr,
      Handler handler)
  {
    service_impl_.async_by_address(impl, h, addr, handler);
  }

  /// Get host information for a named host.
  template <typename Error_Handler>
  void by_name(implementation_type& impl, host& h, const std::string& name,
      Error_Handler error_handler)
  {
    service_impl_.by_name(impl, h, name, error_handler);
  }

  // Asynchronously get host information for a named host.
  template <typename Handler>
  void async_by_name(implementation_type& impl, host& h,
      const std::string& name, Handler handler)
  {
    service_impl_.async_by_name(impl, h, name, handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_HOST_RESOLVER_SERVICE_HPP
