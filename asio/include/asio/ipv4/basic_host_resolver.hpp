//
// basic_host_resolver.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_IPV4_BASIC_HOST_RESOLVER_HPP
#define ASIO_IPV4_BASIC_HOST_RESOLVER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <string>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/default_error_handler.hpp"
#include "asio/service_factory.hpp"
#include "asio/ipv4/address.hpp"
#include "asio/ipv4/host.hpp"

namespace asio {
namespace ipv4 {

/// Implements resolution of host names and addresses.
/**
 * The asio::ipv4::basic_host_resolver class template provides the ability to
 * lookup hosts based on their names or addresses.
 *
 * Most applications will use the asio::ipv4::host_resolver typedef.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
template <typename Service>
class basic_host_resolver
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide host resolution
  /// operations.
  typedef Service service_type;

  /// The native implementation type of the host resolver.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// Construct a basic_host_resolver.
  /**
   * This constructor creates a basic_host_resolver.
   *
   * @param d The demuxer object that the host resolver will use to dispatch
   * handlers for any asynchronous operations.
   */
  explicit basic_host_resolver(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_type::null())
  {
    service_.create(impl_);
  }

  /// Destructor.
  ~basic_host_resolver()
  {
    service_.destroy(impl_);
  }

  /// Get host information for the local machine.
  /**
   * This function is used to obtain host information for the local machine.
   *
   * @param h A host object that receives information about the local machine.
   *
   * @throws asio::error Thrown on failure.
   */
  void get_local_host(host& h)
  {
    service_.get_local_host(impl_, h, default_error_handler());
  }

  /// Get host information for the local machine.
  /**
   * This function is used to obtain host information for the local machine.
   *
   * @param h A host object that receives information assocated with the
   * specified address. After successful completion of this function, the host
   * object is guaranteed to contain at least one address.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void get_local_host(host& h, Error_Handler error_handler)
  {
    service_.get_local_host(impl_, h, error_handler);
  }

  /// Get host information for a specified address.
  /**
   * This function is used to obtain host information assocated with a
   * specified address.
   *
   * @param h A host object that receives information assocated with the
   * specified address. After successful completion of this function, the host
   * object is guaranteed to contain at least one address.
   *
   * @param addr An address object that identifies a host.
   *
   * @throws asio::error Thrown on failure.
   */
  void get_host_by_address(host& h, const address& addr)
  {
    service_.get_host_by_address(impl_, h, addr, default_error_handler());
  }

  /// Get host information for a specified address.
  /**
   * This function is used to obtain host information assocated with a
   * specified address.
   *
   * @param h A host object that receives information assocated with the
   * specified address. After successful completion of this function, the host
   * object is guaranteed to contain at least one address.
   *
   * @param addr An address object that identifies a host.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void get_host_by_address(host& h, const address& addr,
      Error_Handler error_handler)
  {
    service_.get_host_by_address(impl_, h, addr, error_handler);
  }

  /// Get host information for a named host.
  /**
   * This function is used to obtain host information assocated with a
   * specified host name.
   *
   * @param h A host object that receives information assocated with the
   * specified host name.
   *
   * @param name A name that identifies a host.
   *
   * @throws asio::error Thrown on failure.
   */
  void get_host_by_name(host& h, const std::string& name)
  {
    service_.get_host_by_name(impl_, h, name, default_error_handler());
  }

  /// Get host information for a named host.
  /**
   * This function is used to obtain host information assocated with a
   * specified host name.
   *
   * @param h A host object that receives information assocated with the
   * specified host name.
   *
   * @param name A name that identifies a host.
   *
   * @param error_handler The handler to be called when an error occurs. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Error_Handler>
  void get_host_by_name(host& h, const std::string& name,
      Error_Handler error_handler)
  {
    service_.get_host_by_name(impl_, h, name, error_handler);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_BASIC_HOST_RESOLVER_HPP
