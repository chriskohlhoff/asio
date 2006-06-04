//
// basic_resolver.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_RESOLVER_HPP
#define ASIO_BASIC_RESOLVER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_io_object.hpp"
#include "asio/error.hpp"
#include "asio/error_handler.hpp"
#include "asio/resolver_service.hpp"

namespace asio {

/// Provides endpoint resolution functionality.
/**
 * The basic_resolver class template provides the ability to resolve a query
 * to a list of endpoints.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object, Error_Source.
 */
template <typename Protocol, typename Service = resolver_service<Protocol> >
class basic_resolver
  : public basic_io_object<Service>
{
public:
  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  /// The query type.
  typedef typename Protocol::resolver_query query;

  /// The iterator type.
  typedef typename Protocol::resolver_iterator iterator;

  /// The type used for reporting errors.
  typedef asio::error error_type;

  /// Constructor.
  /**
   * This constructor creates a basic_resolver.
   *
   * @param io_service The io_service object that the resolver will use to
   * dispatch handlers for any asynchronous operations performed on the timer.
   */
  explicit basic_resolver(asio::io_service& io_service)
    : basic_io_object<Service>(io_service)
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
    return this->service.cancel(this->implementation);
  }

  /// Resolve a query to a list of entries.
  /**
   * This function is used to resolve a query into a list of endpoint entries.
   *
   * @param q A query object that determines what endpoints will be returned.
   *
   * @returns A forward-only iterator that can be used to traverse the list
   * of endpoint entries.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note A default constructed iterator represents the end of the list.
   *
   * @note A successful call to this function is guaranteed to return at least
   * one entry.
   */
  iterator resolve(const query& q)
  {
    return this->service.resolve(this->implementation, q, throw_error());
  }

  /// Resolve a query to a list of entries.
  /**
   * This function is used to resolve a query into a list of endpoint entries.
   *
   * @param q A query object that determines what endpoints will be returned.
   *
   * @returns A forward-only iterator that can be used to traverse the list
   * of endpoint entries. Returns a default constructed iterator if an error
   * occurs.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation.
   * ); @endcode
   *
   * @note A default constructed iterator represents the end of the list.
   *
   * @note A successful call to this function is guaranteed to return at least
   * one entry.
   */
  template <typename Error_Handler>
  iterator resolve(const query& q, Error_Handler error_handler)
  {
    return this->service.resolve(this->implementation, q, error_handler);
  }

  /// Asynchronously resolve a query to a list of entries.
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
   *   const asio::error& error,   // Result of operation.
   *   resolver::iterator iterator // Forward-only iterator that can be used to
   *                               // traverse the list of endpoint entries.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @note A default constructed iterator represents the end of the list.
   *
   * @note A successful resolve operation is guaranteed to pass at least one
   * entry to the handler.
   */
  template <typename Handler>
  void async_resolve(const query& q, Handler handler)
  {
    return this->service.async_resolve(this->implementation, q, handler);
  }

  /// Resolve an endpoint to a list of entries.
  /**
   * This function is used to resolve an endpoint into a list of endpoint
   * entries.
   *
   * @param e An endpoint object that determines what endpoints will be
   * returned.
   *
   * @returns A forward-only iterator that can be used to traverse the list
   * of endpoint entries.
   *
   * @throws asio::error Thrown on failure.
   *
   * @note A default constructed iterator represents the end of the list.
   *
   * @note A successful call to this function is guaranteed to return at least
   * one entry.
   */
  iterator resolve(const endpoint_type& e)
  {
    return this->service.resolve(this->implementation, e, throw_error());
  }

  /// Resolve an endpoint to a list of entries.
  /**
   * This function is used to resolve an endpoint into a list of endpoint
   * entries.
   *
   * @param e An endpoint object that determines what endpoints will be
   * returned.
   *
   * @returns A forward-only iterator that can be used to traverse the list
   * of endpoint entries. Returns a default constructed iterator if an error
   * occurs.
   *
   * @param error_handler A handler to be called when the operation completes,
   * to indicate whether or not an error has occurred. Copies will be made of
   * the handler as required. The function signature of the handler must be:
   * @code void error_handler(
   *   const asio::error& error // Result of operation.
   * ); @endcode
   *
   * @note A default constructed iterator represents the end of the list.
   *
   * @note A successful call to this function is guaranteed to return at least
   * one entry.
   */
  template <typename Error_Handler>
  iterator resolve(const endpoint_type& e, Error_Handler error_handler)
  {
    return this->service.resolve(this->implementation, e, error_handler);
  }

  /// Asynchronously resolve an endpoint to a list of entries.
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
   *   const asio::error& error,   // Result of operation.
   *   resolver::iterator iterator // Forward-only iterator that can be used to
   *                               // traverse the list of endpoint entries.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
   *
   * @note A default constructed iterator represents the end of the list.
   *
   * @note A successful resolve operation is guaranteed to pass at least one
   * entry to the handler.
   */
  template <typename Handler>
  void async_resolve(const endpoint_type& e, Handler handler)
  {
    return this->service.async_resolve(this->implementation, e, handler);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_RESOLVER_HPP
