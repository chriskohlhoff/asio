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
template <typename Service>
class basic_resolver
  : public basic_io_object<Service>
{
public:
  /// The protocol type.
  typedef typename Service::protocol_type protocol_type;

  /// The endpoint type.
  typedef typename Service::endpoint_type endpoint_type;

  /// The query type.
  typedef typename Service::query_type query;

  /// The iterator type.
  typedef typename Service::iterator_type iterator;

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
  iterator resolve(const query& q)
  {
    return this->service.resolve(this->implementation, q, throw_error());
  }

  /// Resolve a query to a list of entries.
  template <typename Error_Handler>
  iterator resolve(const query& q, Error_Handler error_handler)
  {
    return this->service.resolve(this->implementation, q, error_handler);
  }

  /// Asynchronously resolve a query to a list of entries.
  template <typename Handler>
  void async_resolve(const query& q, Handler handler)
  {
    return this->service.async_resolve(this->implementation, q, handler);
  }

  /// Resolve an endpoint to a list of entries.
  iterator resolve(const endpoint_type& e)
  {
    return this->service.resolve(this->implementation, e, throw_error());
  }

  /// Resolve an endpoint to a list of entries.
  template <typename Error_Handler>
  iterator resolve(const endpoint_type& e, Error_Handler error_handler)
  {
    return this->service.resolve(this->implementation, e, error_handler);
  }

  /// Asynchronously resolve an endpoint to a list of entries.
  template <typename Handler>
  void async_resolve(const endpoint_type& e, Handler handler)
  {
    return this->service.async_resolve(this->implementation, e, handler);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_RESOLVER_HPP
