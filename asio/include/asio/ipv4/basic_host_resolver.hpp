//
// basic_host_resolver.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_BASIC_HOST_RESOLVER_HPP
#define ASIO_IPV4_BASIC_HOST_RESOLVER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <string>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/error_handler.hpp"
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
 *
 * @par Concepts:
 * Async_Object, Error_Source.
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

  /// The type used for reporting errors.
  typedef asio::error error_type;

  /// Construct a basic_host_resolver.
  /**
   * This constructor creates a basic_host_resolver.
   *
   * @param d The demuxer object that the host resolver will use to dispatch
   * handlers for any asynchronous operations.
   */
  explicit basic_host_resolver(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.open(impl_);
  }

  /// Destructor.
  ~basic_host_resolver()
  {
    service_.close(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the host
   * resolver uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the demuxer object that host resolver will use to
   * dispatch handlers. Ownership is not transferred to the caller.
   */
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Open the host resolver.
  void open()
  {
    service_.open(impl_);
  }

  /// Close the host resolver.
  /**
   * This function is used to close the host resolver. Any asynchronous
   * operations will be cancelled immediately.
   */
  void close()
  {
    service_.close(impl_);
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
    service_.get_local_host(impl_, h, throw_error());
  }

  /// Get host information for the local machine.
  /**
   * This function is used to obtain host information for the local machine.
   *
   * @param h A host object that receives information associated with the
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
   * This function is used to obtain host information associated with a
   * specified address.
   *
   * @param h A host object that receives information associated with the
   * specified address. After successful completion of this function, the host
   * object is guaranteed to contain at least one address.
   *
   * @param addr An address object that identifies a host.
   *
   * @throws asio::error Thrown on failure.
   */
  void get_host_by_address(host& h, const address& addr)
  {
    service_.get_host_by_address(impl_, h, addr, throw_error());
  }

  /// Get host information for a specified address.
  /**
   * This function is used to obtain host information associated with a
   * specified address.
   *
   * @param h A host object that receives information associated with the
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

  /// Asynchronously get host information for a specified address.
  /**
   * This function is used to asynchronously obtain host information associated
   * with a specified address. The function call always returns immediately.
   *
   * @param h A host object that receives information associated with the
   * specified address. After successful completion of the asynchronous
   * operation, the host object is guaranteed to contain at least one address.
   * Ownership of the host object is retained by the caller, which must
   * guarantee that it is valid until the handler is called.
   *
   * @param addr An address object that identifies a host. Copies will be made
   * of the address object as required.
   *
   * @param handler The handler to be called when the resolve operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Handler>
  void async_get_host_by_address(host& h, const address& addr, Handler handler)
  {
    service_.async_get_host_by_address(impl_, h, addr, handler);
  }

  /// Get host information for a named host.
  /**
   * This function is used to obtain host information associated with a
   * specified host name.
   *
   * @param h A host object that receives information associated with the
   * specified host name.
   *
   * @param name A name that identifies a host.
   *
   * @throws asio::error Thrown on failure.
   */
  void get_host_by_name(host& h, const std::string& name)
  {
    service_.get_host_by_name(impl_, h, name, throw_error());
  }

  /// Get host information for a named host.
  /**
   * This function is used to obtain host information associated with a
   * specified host name.
   *
   * @param h A host object that receives information associated with the
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

  /// Asynchronously get host information for a named host.
  /**
   * This function is used to asynchronously obtain host information associated
   * with a specified host name. The function call always returns immediately.
   *
   * @param h A host object that receives information associated with the
   * specified address. After successful completion of the asynchronous
   * operation, the host object is guaranteed to contain at least one address.
   * Ownership of the host object is retained by the caller, which must
   * guarantee that it is valid until the handler is called.
   *
   * @param name A name that identifies a host. Copies will be made of the name
   * as required.
   *
   * @param handler The handler to be called when the resolve operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Handler>
  void async_get_host_by_name(host& h, const std::string& name, Handler handler)
  {
    service_.async_get_host_by_name(impl_, h, name, handler);
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
