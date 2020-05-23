//
// generic/basic_endpoint.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_GENERIC_BASIC_ENDPOINT_HPP
#define ASIO_GENERIC_BASIC_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
# include "asio/detail/apple_nw_ptr.hpp"
# include <Network/Network.h>
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
# include "asio/generic/detail/endpoint.hpp"
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace generic {

/// Describes an endpoint for any socket type.
/**
 * The asio::generic::basic_endpoint class template describes an endpoint
 * that may be associated with any socket type.
 *
 * @note The socket types sockaddr type must be able to fit into a
 * @c sockaddr_storage structure.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Endpoint.
 */
template <typename Protocol>
class basic_endpoint
{
public:
  /// The protocol type associated with the endpoint.
  typedef Protocol protocol_type;

  /// The type of the endpoint structure. This type is dependent on the
  /// underlying implementation of the socket layer.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined data_type;
#elif !defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  typedef asio::detail::socket_addr_type data_type;
#endif

  /// Default constructor.
  basic_endpoint() ASIO_NOEXCEPT
#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    : endpoint_(),
      protocol_(asio::detail::apple_nw_ptr<nw_parameters_t>(), 0)
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  {
  }

#if !defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  /// Construct an endpoint from the specified socket address.
  basic_endpoint(const void* socket_address,
      std::size_t socket_address_size, int socket_protocol = 0)
    : impl_(socket_address, socket_address_size, socket_protocol)
  {
  }
#endif // !defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

  /// Construct an endpoint from the specific endpoint type.
  template <typename Endpoint>
  basic_endpoint(const Endpoint& endpoint)
#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    : endpoint_(endpoint.apple_nw_create_endpoint()),
      protocol_(endpoint.protocol())
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    : impl_(endpoint.data(), endpoint.size(), endpoint.protocol().protocol())
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  {
  }

  /// Copy constructor.
  basic_endpoint(const basic_endpoint& other)
#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    : endpoint_(other.endpoint_),
      protocol_(other.protocol_)
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    : impl_(other.impl_)
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  {
  }

#if defined(ASIO_HAS_MOVE)
  /// Move constructor.
  basic_endpoint(basic_endpoint&& other)
#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    : endpoint_(ASIO_MOVE_CAST(
          asio::detail::apple_nw_ptr<nw_endpoint_t>)(
            other.endpoint_)),
      protocol_(ASIO_MOVE_CAST(protocol_type)(other.protocol_))
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    : impl_(other.impl_)
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  {
  }
#endif // defined(ASIO_HAS_MOVE)

  /// Assign from another endpoint.
  basic_endpoint& operator=(const basic_endpoint& other)
  {
#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    endpoint_ = other.endpoint_;
    protocol_ = other.protocol_;
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    impl_ = other.impl_;
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    return *this;
  }

#if defined(ASIO_HAS_MOVE)
  /// Move-assign from another endpoint.
  basic_endpoint& operator=(basic_endpoint&& other)
  {
#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    endpoint_ = ASIO_MOVE_CAST(
        asio::detail::apple_nw_ptr<nw_endpoint_t>)(
          other.endpoint_);
    protocol_ = ASIO_MOVE_CAST(protocol_type)(other.protocol_);
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    impl_ = other.impl_;
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE)

  /// The protocol associated with the endpoint.
  protocol_type protocol() const
  {
#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    return protocol_;
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    return protocol_type(impl_.family(), impl_.protocol());
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  }

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  // The following functions comprise the extensible interface for the Endpoint
  // concept when targeting the Apple Network Framework.

  // Create a new native object corresponding to the endpoint.
  asio::detail::apple_nw_ptr<nw_endpoint_t>
  apple_nw_create_endpoint() const
  {
    return endpoint_;
  }

  // Set the endpoint from the native object.
  void apple_nw_set_endpoint(
      asio::detail::apple_nw_ptr<nw_endpoint_t> new_ep)
  {
    endpoint_ = new_ep;
  }

  // Set the protocol.
  void apple_nw_set_protocol(protocol_type new_protocol)
  {
    protocol_ = new_protocol;
  }
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  /// Get the underlying endpoint in the native type.
  data_type* data()
  {
    return impl_.data();
  }

  /// Get the underlying endpoint in the native type.
  const data_type* data() const
  {
    return impl_.data();
  }

  /// Get the underlying size of the endpoint in the native type.
  std::size_t size() const
  {
    return impl_.size();
  }

  /// Set the underlying size of the endpoint in the native type.
  void resize(std::size_t new_size)
  {
    impl_.resize(new_size);
  }

  /// Get the capacity of the endpoint in the native type.
  std::size_t capacity() const
  {
    return impl_.capacity();
  }
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

  /// Compare two endpoints for equality.
  friend bool operator==(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    return e1.endpoint_ == e2.endpoint_ && e1.protocol_ == e2.protocol_;
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
    return e1.impl_ == e2.impl_;
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  }

  /// Compare two endpoints for inequality.
  friend bool operator!=(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return !(e1 == e2);
  }

#if !defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  /// Compare endpoints for ordering.
  friend bool operator<(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return e1.impl_ < e2.impl_;
  }

  /// Compare endpoints for ordering.
  friend bool operator>(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return e2.impl_ < e1.impl_;
  }

  /// Compare endpoints for ordering.
  friend bool operator<=(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return !(e2 < e1);
  }

  /// Compare endpoints for ordering.
  friend bool operator>=(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return !(e1 < e2);
  }
#endif // !defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

private:
#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  // The underlying native endpoint.
  asio::detail::apple_nw_ptr<nw_endpoint_t> endpoint_;

  // The associated protocol object.
  Protocol protocol_;
#else // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
  // The underlying generic endpoint.
  asio::generic::detail::endpoint impl_;
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
};

} // namespace generic
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_GENERIC_BASIC_ENDPOINT_HPP
