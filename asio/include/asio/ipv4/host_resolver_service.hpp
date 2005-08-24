//
// host_resolver_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
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

#include "asio/detail/push_options.hpp"
#include <cstring>
#include <memory>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/error.hpp"
#include "asio/ipv4/detail/host_resolver_service.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/ipv4/host.hpp"

namespace asio {
namespace ipv4 {

template <typename Allocator = std::allocator<void> >
class host_resolver_service
  : private boost::noncopyable
{
public:
  /// The demuxer type for this service.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

private:
  // The type of the platform-specific implementation.
  typedef detail::host_resolver_service<demuxer_type> service_impl_type;

public:
  /// The type of the host resolver.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined impl_type;
#else
  typedef typename service_impl_type::impl_type impl_type;
#endif

  /// Constructor.
  host_resolver_service(demuxer_type& demuxer)
    : service_impl_(demuxer.get_service(service_factory<service_impl_type>()))
  {
  }

  /// Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return service_impl_.demuxer();
  }

  /// Return a null host resolver implementation.
  impl_type null() const
  {
    return service_impl_.null();
  }

  /// Create a new host resolver implementation.
  void create(impl_type& impl)
  {
    service_impl_.create(impl);
  }

  /// Destroy a host resolver implementation.
  void destroy(impl_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Get host information for the local machine.
  template <typename Error_Handler>
  void get_local_host(impl_type& impl, host& h, Error_Handler error_handler)
  {
    service_impl_.get_local_host(impl, h, error_handler);
  }

  /// Get host information for a specified address.
  template <typename Error_Handler>
  void get_host_by_address(impl_type& impl, host& h, const address& addr,
      Error_Handler error_handler)
  {
    service_impl_.get_host_by_address(impl, h, addr, error_handler);
  }

  /// Get host information for a named host.
  template <typename Error_Handler>
  void get_host_by_name(impl_type& impl, host& h, const std::string& name,
      Error_Handler error_handler)
  {
    service_impl_.get_host_by_name(impl, h, name, error_handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_HOST_RESOLVER_SERVICE_HPP
