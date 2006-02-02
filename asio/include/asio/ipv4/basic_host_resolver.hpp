//
// basic_host_resolver.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include "asio/detail/pop_options.hpp"

#include "asio/basic_io_object.hpp"
#include "asio/error.hpp"
#include "asio/error_handler.hpp"
#include "asio/ipv4/address.hpp"
#include "asio/ipv4/host.hpp"

namespace asio {
namespace ipv4 {

/// Implements resolution of host names and addresses.
/**
 * The asio::ipv4::basic_host_resolver class template provides the
 * ability to lookup hosts based on their names or addresses.
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
  : public basic_io_object<Service>
{
public:
  /// The io_service type for this I/O object.
  typedef typename Service::io_service_type io_service_type;

  /// The type used for reporting errors.
  typedef asio::error error_type;

  /// Construct a basic_host_resolver.
  /**
   * This constructor creates a basic_host_resolver.
   *
   * @param io_service The io_service object that the host resolver will use to
   * dispatch handlers for any asynchronous operations.
   */
  explicit basic_host_resolver(io_service_type& io_service)
    : basic_io_object<Service>(io_service)
  {
  }

  /// Cancel any asynchronous operations on the host resolver.
  /**
   * This function forces the completion of any pending asynchronous
   * operations on the host resolver. The handler for each cancelled operation
   * will be invoked with the asio::error::operation_aborted error code.
   */
  void cancel()
  {
    this->service.cancel(this->implementation);
  }

  /// Get host information for the local machine.
  /**
   * This function is used to obtain host information for the local machine.
   *
   * @param h A host object that receives information about the local machine.
   *
   * @throws asio::error Thrown on failure.
   *
   * @par Example:
   * @code
   * asio::ipv4::host_resolver resolver(io_service);
   * ...
   * asio::ipv4::host host;
   * resolver.local(host);
   * std::cout << "Name: " << host.name();
   * std::cout << "Address: " << host.addresses(0);
   * @endcode
   */
  void local(host& h)
  {
    this->service.local(this->implementation, h, throw_error());
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
   * will be made of the handler as required. The function signature of the
   * handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @par Example:
   * @code
   * asio::ipv4::host_resolver resolver(io_service);
   * ...
   * asio::ipv4::host host;
   * asio::error error;
   * resolver.local(host, asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * @endcode
   */
  template <typename Error_Handler>
  void local(host& h, Error_Handler error_handler)
  {
    this->service.local(this->implementation, h, error_handler);
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
   *
   * @par Example:
   * @code
   * asio::ipv4::host_resolver resolver(io_service);
   * ...
   * asio::ipv4::host host;
   * asio::ipv4::address address("1.2.3.4");
   * resolver.by_address(host, address);
   * std::cout << "Name: " << host.name();
   * @endcode
   */
  void by_address(host& h, const address& addr)
  {
    this->service.by_address(this->implementation, h, addr, throw_error());
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
   * will be made of the handler as required. The function signature of the
   * handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @par Example:
   * @code
   * asio::ipv4::host_resolver resolver(io_service);
   * ...
   * asio::ipv4::host host;
   * asio::ipv4::address address("1.2.3.4");
   * asio::error error;
   * resolver.by_address(host, address,
   *     asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * std::cout << "Name: " << host.name();
   * @endcode
   */
  template <typename Error_Handler>
  void by_address(host& h, const address& addr,
      Error_Handler error_handler)
  {
    this->service.by_address(this->implementation, h, addr, error_handler);
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
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Handler>
  void async_by_address(host& h, const address& addr, Handler handler)
  {
    this->service.async_by_address(this->implementation, h, addr, handler);
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
   *
   * @par Example:
   * @code
   * asio::ipv4::host_resolver resolver(io_service);
   * ...
   * asio::ipv4::host host;
   * std::string name("myhost");
   * resolver.by_name(host, name);
   * std::cout << "Address: " << host.addresses(0);
   * @endcode
   */
  void by_name(host& h, const std::string& name)
  {
    this->service.by_name(this->implementation, h, name, throw_error());
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
   * will be made of the handler as required. The function signature of the
   * handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   *
   * @par Example:
   * @code
   * asio::ipv4::host_resolver resolver(io_service);
   * ...
   * asio::ipv4::host host;
   * std::string name("myhost");
   * asio::error error;
   * resolver.by_name(host, name, asio::assign_error(error));
   * if (error)
   * {
   *   // An error occurred.
   * }
   * std::cout << "Address: " << host.addresses(0);
   * @endcode
   */
  template <typename Error_Handler>
  void by_name(host& h, const std::string& name,
      Error_Handler error_handler)
  {
    this->service.by_name(this->implementation, h, name, error_handler);
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
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Handler>
  void async_by_name(host& h, const std::string& name, Handler handler)
  {
    this->service.async_by_name(this->implementation, h, name, handler);
  }
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_BASIC_HOST_RESOLVER_HPP
