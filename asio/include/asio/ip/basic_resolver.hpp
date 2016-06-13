//
// ip/basic_resolver.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_BASIC_RESOLVER_HPP
#define ASIO_IP_BASIC_RESOLVER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "../detail/config.hpp"
#include <string>
#include "../basic_io_object.hpp"
#include "../detail/handler_type_requirements.hpp"
#include "../detail/throw_error.hpp"
#include "../error.hpp"
#include "../ip/basic_resolver_iterator.hpp"
#include "../ip/basic_resolver_query.hpp"
#include "../ip/basic_resolver_results.hpp"
#include "../ip/resolver_base.hpp"
#include "../ip/resolver_service.hpp"

#include "../detail/push_options.hpp"

namespace asio {
namespace ip {

/// Provides endpoint resolution functionality.
/**
 * The basic_resolver class template provides the ability to resolve a query
 * to a list of endpoints.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
template <typename InternetProtocol,
    typename ResolverService = resolver_service<InternetProtocol> >
class basic_resolver
  : public basic_io_object<ResolverService>,
    public resolver_base
{
public:
  /// The protocol type.
  typedef InternetProtocol protocol_type;

  /// The endpoint type.
  typedef typename InternetProtocol::endpoint endpoint_type;

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated.) The query type.
  typedef basic_resolver_query<InternetProtocol> query;

  /// (Deprecated.) The iterator type.
  typedef basic_resolver_iterator<InternetProtocol> iterator;
#endif // !defined(ASIO_NO_DEPRECATED)

  /// The results type.
  typedef basic_resolver_results<InternetProtocol> results_type;

  /// Constructor.
  /**
   * This constructor creates a basic_resolver.
   *
   * @param io_context The io_context object that the resolver will use to
   * dispatch handlers for any asynchronous operations performed on the timer.
   */
  explicit basic_resolver(asio::io_context& io_context)
    : basic_io_object<ResolverService>(io_context)
  {
  }

  /// Cancel any asynchronous operations that are waiting on the resolver.
  /**
   * This function forces the completion of any pending asynchronous
   * operations on the host resolver. The handler for each cancelled operation
   * will be invoked with the asio::error::operation_aborted error code.
   */
  void cancel()
  {
    return this->get_service().cancel(this->get_implementation());
  }

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated.) Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve a query into a list of endpoint entries.
   *
   * @param q A query object that determines what endpoints will be returned.
   *
   * @returns A range object representing the list of endpoint entries. A
   * successful call to this function is guaranteed to return a non-empty
   * range.
   *
   * @throws asio::system_error Thrown on failure.
   */
  results_type resolve(const query& q)
  {
    asio::error_code ec;
    results_type r = this->get_service().resolve(
        this->get_implementation(), q, ec);
    asio::detail::throw_error(ec, "resolve");
    return r;
  }

  /// (Deprecated.) Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve a query into a list of endpoint entries.
   *
   * @param q A query object that determines what endpoints will be returned.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns A range object representing the list of endpoint entries. An
   * empty range is returned if an error occurs. A successful call to this
   * function is guaranteed to return a non-empty range.
   */
  results_type resolve(const query& q, asio::error_code& ec)
  {
    return this->get_service().resolve(this->get_implementation(), q, ec);
  }
#endif // !defined(ASIO_NO_DEPRECATED)

  /// Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @returns A range object representing the list of endpoint entries. A
   * successful call to this function is guaranteed to return a non-empty
   * range.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  results_type resolve(const std::string& host, const std::string& service)
  {
    return resolve(host, service, resolver_base::flags());
  }

  /// Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns A range object representing the list of endpoint entries. An
   * empty range is returned if an error occurs. A successful call to this
   * function is guaranteed to return a non-empty range.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  results_type resolve(const std::string& host, const std::string& service,
      asio::error_code& ec)
  {
    return resolve(host, service, resolver_base::flags(), ec);
  }

  /// Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param resolve_flags A set of flags that determine how name resolution
   * should be performed. The default flags are suitable for communication with
   * remote hosts.
   *
   * @returns A range object representing the list of endpoint entries. A
   * successful call to this function is guaranteed to return a non-empty
   * range.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  results_type resolve(const std::string& host, const std::string& service,
      resolver_base::flags resolve_flags)
  {
    asio::error_code ec;
    basic_resolver_query<protocol_type> q(host, service, resolve_flags);
    results_type r = this->get_service().resolve(
        this->get_implementation(), q, ec);
    asio::detail::throw_error(ec, "resolve");
    return r;
  }

  /// Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param resolve_flags A set of flags that determine how name resolution
   * should be performed. The default flags are suitable for communication with
   * remote hosts.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns A range object representing the list of endpoint entries. An
   * empty range is returned if an error occurs. A successful call to this
   * function is guaranteed to return a non-empty range.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  results_type resolve(const std::string& host, const std::string& service,
      resolver_base::flags resolve_flags, asio::error_code& ec)
  {
    basic_resolver_query<protocol_type> q(host, service, resolve_flags);
    return this->get_service().resolve(this->get_implementation(), q, ec);
  }

  /// Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param protocol A protocol object, normally representing either the IPv4 or
   * IPv6 version of an internet protocol.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @returns A range object representing the list of endpoint entries. A
   * successful call to this function is guaranteed to return a non-empty
   * range.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  results_type resolve(const protocol_type& protocol, const std::string& host,
      const std::string& service)
  {
    return resolve(protocol, host, service, resolver_base::flags());
  }

  /// Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param protocol A protocol object, normally representing either the IPv4 or
   * IPv6 version of an internet protocol.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns A range object representing the list of endpoint entries. An
   * empty range is returned if an error occurs. A successful call to this
   * function is guaranteed to return a non-empty range.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  results_type resolve(const protocol_type& protocol, const std::string& host,
      const std::string& service, asio::error_code& ec)
  {
    return resolve(protocol, host, service, resolver_base::flags(), ec);
  }

  /// Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param protocol A protocol object, normally representing either the IPv4 or
   * IPv6 version of an internet protocol.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param resolve_flags A set of flags that determine how name resolution
   * should be performed. The default flags are suitable for communication with
   * remote hosts.
   *
   * @returns A range object representing the list of endpoint entries. A
   * successful call to this function is guaranteed to return a non-empty
   * range.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  results_type resolve(const protocol_type& protocol, const std::string& host,
      const std::string& service, resolver_base::flags resolve_flags)
  {
    asio::error_code ec;
    basic_resolver_query<protocol_type> q(
        protocol, host, service, resolve_flags);
    results_type r = this->get_service().resolve(
        this->get_implementation(), q, ec);
    asio::detail::throw_error(ec, "resolve");
    return r;
  }

  /// Perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param protocol A protocol object, normally representing either the IPv4 or
   * IPv6 version of an internet protocol.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param resolve_flags A set of flags that determine how name resolution
   * should be performed. The default flags are suitable for communication with
   * remote hosts.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns A range object representing the list of endpoint entries. An
   * empty range is returned if an error occurs. A successful call to this
   * function is guaranteed to return a non-empty range.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  results_type resolve(const protocol_type& protocol, const std::string& host,
      const std::string& service, resolver_base::flags resolve_flags,
      asio::error_code& ec)
  {
    basic_resolver_query<protocol_type> q(
        protocol, host, service, resolve_flags);
    return this->get_service().resolve(this->get_implementation(), q, ec);
  }

#if !defined(ASIO_NO_DEPRECATED)
  /// (Deprecated.) Asynchronously perform forward resolution of a query to a
  /// list of entries.
  /**
   * This function is used to asynchronously resolve a query into a list of
   * endpoint entries.
   *
   * @param q A query object that determines what endpoints will be returned.
   *
   * @param handler The handler to be called when the resolve operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   resolver::results_type results // Resolved endpoints as a range.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   *
   * A successful resolve operation is guaranteed to pass a non-empty range to
   * the handler.
   */
  template <typename ResolveHandler>
  ASIO_INITFN_RESULT_TYPE(ResolveHandler,
      void (asio::error_code, results_type))
  async_resolve(const query& q,
      ASIO_MOVE_ARG(ResolveHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a ResolveHandler.
    ASIO_RESOLVE_HANDLER_CHECK(
        ResolveHandler, handler, results_type) type_check;

    return this->get_service().async_resolve(this->get_implementation(), q,
        ASIO_MOVE_CAST(ResolveHandler)(handler));
  }
#endif // !defined(ASIO_NO_DEPRECATED)

  /// Asynchronously perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param handler The handler to be called when the resolve operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   resolver::results_type results // Resolved endpoints as a range.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   *
   * A successful resolve operation is guaranteed to pass a non-empty range to
   * the handler.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  template <typename ResolveHandler>
  ASIO_INITFN_RESULT_TYPE(ResolveHandler,
      void (asio::error_code, results_type))
  async_resolve(const std::string& host, const std::string& service,
      ASIO_MOVE_ARG(ResolveHandler) handler)
  {
    return async_resolve(host, service, resolver_base::flags(),
        ASIO_MOVE_CAST(ResolveHandler)(handler));
  }

  /// Asynchronously perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param resolve_flags A set of flags that determine how name resolution
   * should be performed. The default flags are suitable for communication with
   * remote hosts.
   *
   * @param handler The handler to be called when the resolve operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   resolver::results_type results // Resolved endpoints as a range.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   *
   * A successful resolve operation is guaranteed to pass a non-empty range to
   * the handler.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  template <typename ResolveHandler>
  ASIO_INITFN_RESULT_TYPE(ResolveHandler,
      void (asio::error_code, results_type))
  async_resolve(const std::string& host, const std::string& service,
      resolver_base::flags resolve_flags,
      ASIO_MOVE_ARG(ResolveHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a ResolveHandler.
    ASIO_RESOLVE_HANDLER_CHECK(
        ResolveHandler, handler, results_type) type_check;

    basic_resolver_query<protocol_type> q(host, service, resolve_flags);
    return this->get_service().async_resolve(this->get_implementation(), q,
        ASIO_MOVE_CAST(ResolveHandler)(handler));
  }

  /// Asynchronously perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param protocol A protocol object, normally representing either the IPv4 or
   * IPv6 version of an internet protocol.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param handler The handler to be called when the resolve operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   resolver::results_type results // Resolved endpoints as a range.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   *
   * A successful resolve operation is guaranteed to pass a non-empty range to
   * the handler.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  template <typename ResolveHandler>
  ASIO_INITFN_RESULT_TYPE(ResolveHandler,
      void (asio::error_code, results_type))
  async_resolve(const protocol_type& protocol, const std::string& host,
      const std::string& service, ASIO_MOVE_ARG(ResolveHandler) handler)
  {
    return async_resolve(protocol, host, service, resolver_base::flags(),
        ASIO_MOVE_CAST(ResolveHandler)(handler));
  }

  /// Asynchronously perform forward resolution of a query to a list of entries.
  /**
   * This function is used to resolve host and service names into a list of
   * endpoint entries.
   *
   * @param protocol A protocol object, normally representing either the IPv4 or
   * IPv6 version of an internet protocol.
   *
   * @param host A string identifying a location. May be a descriptive name or
   * a numeric address string. If an empty string and the passive flag has been
   * specified, the resolved endpoints are suitable for local service binding.
   * If an empty string and passive is not specified, the resolved endpoints
   * will use the loopback address.
   *
   * @param service A string identifying the requested service. This may be a
   * descriptive name or a numeric string corresponding to a port number. May
   * be an empty string, in which case all resolved endpoints will have a port
   * number of 0.
   *
   * @param resolve_flags A set of flags that determine how name resolution
   * should be performed. The default flags are suitable for communication with
   * remote hosts.
   *
   * @param handler The handler to be called when the resolve operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   resolver::results_type results // Resolved endpoints as a range.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   *
   * A successful resolve operation is guaranteed to pass a non-empty range to
   * the handler.
   *
   * @note On POSIX systems, host names may be locally defined in the file
   * <tt>/etc/hosts</tt>. On Windows, host names may be defined in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\hosts</tt>. Remote host name
   * resolution is performed using DNS. Operating systems may use additional
   * locations when resolving host names (such as NETBIOS names on Windows).
   *
   * On POSIX systems, service names are typically defined in the file
   * <tt>/etc/services</tt>. On Windows, service names may be found in the file
   * <tt>c:\\windows\\system32\\drivers\\etc\\services</tt>. Operating systems
   * may use additional locations when resolving service names.
   */
  template <typename ResolveHandler>
  ASIO_INITFN_RESULT_TYPE(ResolveHandler,
      void (asio::error_code, results_type))
  async_resolve(const protocol_type& protocol, const std::string& host,
      const std::string& service, resolver_base::flags resolve_flags,
      ASIO_MOVE_ARG(ResolveHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a ResolveHandler.
    ASIO_RESOLVE_HANDLER_CHECK(
        ResolveHandler, handler, results_type) type_check;

    basic_resolver_query<protocol_type> q(
        protocol, host, service, resolve_flags);
    return this->get_service().async_resolve(this->get_implementation(), q,
        ASIO_MOVE_CAST(ResolveHandler)(handler));
  }

  /// Perform reverse resolution of an endpoint to a list of entries.
  /**
   * This function is used to resolve an endpoint into a list of endpoint
   * entries.
   *
   * @param e An endpoint object that determines what endpoints will be
   * returned.
   *
   * @returns A range object representing the list of endpoint entries. A
   * successful call to this function is guaranteed to return a non-empty
   * range.
   *
   * @throws asio::system_error Thrown on failure.
   */
  results_type resolve(const endpoint_type& e)
  {
    asio::error_code ec;
    results_type i = this->get_service().resolve(
        this->get_implementation(), e, ec);
    asio::detail::throw_error(ec, "resolve");
    return i;
  }

  /// Perform reverse resolution of an endpoint to a list of entries.
  /**
   * This function is used to resolve an endpoint into a list of endpoint
   * entries.
   *
   * @param e An endpoint object that determines what endpoints will be
   * returned.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns A range object representing the list of endpoint entries. An
   * empty range is returned if an error occurs. A successful call to this
   * function is guaranteed to return a non-empty range.
   */
  results_type resolve(const endpoint_type& e, asio::error_code& ec)
  {
    return this->get_service().resolve(this->get_implementation(), e, ec);
  }

  /// Asynchronously perform reverse resolution of an endpoint to a list of
  /// entries.
  /**
   * This function is used to asynchronously resolve an endpoint into a list of
   * endpoint entries.
   *
   * @param e An endpoint object that determines what endpoints will be
   * returned.
   *
   * @param handler The handler to be called when the resolve operation
   * completes. Copies will be made of the handler as required. The function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   resolver::results_type results // Resolved endpoints as a range.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_context::post().
   *
   * A successful resolve operation is guaranteed to pass a non-empty range to
   * the handler.
   */
  template <typename ResolveHandler>
  ASIO_INITFN_RESULT_TYPE(ResolveHandler,
      void (asio::error_code, results_type))
  async_resolve(const endpoint_type& e,
      ASIO_MOVE_ARG(ResolveHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a ResolveHandler.
    ASIO_RESOLVE_HANDLER_CHECK(
        ResolveHandler, handler, results_type) type_check;

    return this->get_service().async_resolve(this->get_implementation(), e,
        ASIO_MOVE_CAST(ResolveHandler)(handler));
  }
};

} // namespace ip
} // namespace asio

#include "../detail/pop_options.hpp"

#endif // ASIO_IP_BASIC_RESOLVER_HPP
