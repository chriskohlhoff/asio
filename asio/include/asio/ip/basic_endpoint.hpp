//
// basic_endpoint.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_BASIC_ENDPOINT_HPP
#define ASIO_IP_BASIC_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include <boost/detail/workaround.hpp>
#include <cstring>
#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
# include <iostream>
#endif // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/ip/address.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ip {

/// Describes an endpoint for a version-independent IP socket.
/**
 * The asio::ip::basic_endpoint class template describes an endpoint that may
 * be associated with a particular socket.
 *
 * @par Thread Safety:
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

  /// The IPv4 endpoint type.
  typedef typename Protocol::ipv4_protocol::endpoint ipv4_endpoint;

  /// The IPv6 endpoint type.
  typedef typename Protocol::ipv6_protocol::endpoint ipv6_endpoint;

  /// The type of the endpoint structure. This type is dependent on the
  /// underlying implementation of the socket layer.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined data_type;
#else
  typedef asio::detail::socket_addr_type data_type;
#endif

  /// The type for the size of the endpoint structure. This type is dependent on
  /// the underlying implementation of the socket layer.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined size_type;
#else
  typedef asio::detail::socket_addr_len_type size_type;
#endif

  /// Default constructor.
  basic_endpoint()
  {
    using namespace std; // For memcpy and memset.
    ipv4_endpoint endpoint;
    memcpy(&data_, endpoint.data(), endpoint.size());
  }

  /// Construct an endpoint using a port number, specified in the host's byte
  /// order. The IP address will be the any address (i.e. in6addr_any). This
  /// constructor would typically be used for accepting new connections.
  basic_endpoint(unsigned short port_num, const Protocol& protocol)
  {
    using namespace std; // For memcpy.
    if (protocol.family() == PF_INET)
    {
      ipv4_endpoint endpoint(port_num);
      memcpy(&data_, endpoint.data(), endpoint.size());
    }
    else
    {
      ipv6_endpoint endpoint(port_num);
      memcpy(&data_, endpoint.data(), endpoint.size());
    }
  }

  /// Construct an endpoint using a port number and an IP address. This
  /// constructor may be used for accepting connections on a specific interface
  /// or for making a connection to a remote endpoint.
  basic_endpoint(unsigned short port_num, const asio::ip::address& addr)
  {
    using namespace std; // For memcpy.
    if (addr.is_ipv4())
    {
      ipv4_endpoint endpoint(port_num, addr.to_ipv4());
      memcpy(&data_, endpoint.data(), endpoint.size());
    }
    else
    {
      ipv6_endpoint endpoint(port_num, addr.to_ipv6());
      memcpy(&data_, endpoint.data(), endpoint.size());
    }
  }

  /// Construct an endpoint from an IPv4 endpoint.
  basic_endpoint(const ipv4_endpoint& endpoint)
  {
    using namespace std; // For memcpy.
    memcpy(&data_, endpoint.data(), endpoint.size());
  }

  /// Construct an endpoint from an IPv6 endpoint.
  basic_endpoint(const ipv6_endpoint& endpoint)
  {
    using namespace std; // For memcpy.
    memcpy(&data_, endpoint.data(), endpoint.size());
  }

  /// Copy constructor.
  basic_endpoint(const basic_endpoint& other)
    : data_(other.data_)
  {
  }

  /// Assign from another endpoint.
  basic_endpoint& operator=(const basic_endpoint& other)
  {
    data_ = other.data_;
    return *this;
  }

  /// The protocol associated with the endpoint.
  protocol_type protocol() const
  {
    if (data_.ss_family == AF_INET)
    {
      typename Protocol::ipv4_protocol ipv4_protocol;
      return Protocol(ipv4_protocol);
    }
    typename Protocol::ipv6_protocol ipv6_protocol;
    return Protocol(ipv6_protocol);
  }

  /// Get the underlying endpoint in the native type.
  data_type* data()
  {
    return reinterpret_cast<data_type*>(&data_);
  }

  /// Get the underlying endpoint in the native type.
  const data_type* data() const
  {
    return reinterpret_cast<const data_type*>(&data_);
  }

  /// Get the underlying size of the endpoint in the native type.
  size_type size() const
  {
    if (data_.ss_family == AF_INET)
      return sizeof(asio::detail::inet_addr_v4_type);
    else
      return sizeof(asio::detail::inet_addr_v6_type);
  }

  /// Set the underlying size of the endpoint in the native type.
  void resize(size_type size)
  {
    if (size > sizeof(data_))
    {
      asio::error e(asio::error::invalid_argument);
      boost::throw_exception(e);
    }
  }

  /// Get the capacity of the endpoint in the native type.
  size_type capacity() const
  {
    return sizeof(data_);
  }

  /// Get the port associated with the endpoint. The port number is always in
  /// the host's byte order.
  unsigned short port() const
  {
    if (data_.ss_family == AF_INET)
    {
      return asio::detail::socket_ops::network_to_host_short(
          reinterpret_cast<const asio::detail::inet_addr_v4_type&>(
            data_).sin_port);
    }
    else
    {
      return asio::detail::socket_ops::network_to_host_short(
          reinterpret_cast<const asio::detail::inet_addr_v6_type&>(
            data_).sin6_port);
    }
  }

  /// Set the port associated with the endpoint. The port number is always in
  /// the host's byte order.
  void port(unsigned short port_num)
  {
    if (data_.ss_family == AF_INET)
    {
      reinterpret_cast<asio::detail::inet_addr_v4_type&>(data_).sin_port
        = asio::detail::socket_ops::host_to_network_short(port_num);
    }
    else
    {
      reinterpret_cast<asio::detail::inet_addr_v6_type&>(data_).sin6_port
        = asio::detail::socket_ops::host_to_network_short(port_num);
    }
  }

  /// Get the IP address associated with the endpoint.
  asio::ip::address address() const
  {
    using namespace std; // For memcpy.
    if (data_.ss_family == AF_INET)
    {
      ipv4_endpoint endpoint;
      memcpy(endpoint.data(), &data_, endpoint.size());
      return endpoint.address();
    }
    else
    {
      ipv6_endpoint endpoint;
      memcpy(endpoint.data(), &data_, endpoint.size());
      return endpoint.address();
    }
  }

  /// Set the IP address associated with the endpoint.
  void address(const asio::ip::address& addr)
  {
    basic_endpoint<Protocol> tmp_endpoint(port(), addr);
    data_ = tmp_endpoint.data_;
  }

  /// Compare two endpoints for equality.
  friend bool operator==(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return e1.address() == e2.address() && e1.port() == e2.port();
  }

  /// Compare two endpoints for inequality.
  friend bool operator!=(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    return e1.address() != e2.address() || e1.port() != e2.port();
  }

  /// Compare endpoints for ordering.
  friend bool operator<(const basic_endpoint<Protocol>& e1,
      const basic_endpoint<Protocol>& e2)
  {
    if (e1.address() < e2.address())
      return true;
    if (e1.address() != e2.address())
      return false;
    return e1.port() < e2.port();
  }

private:
  // The underlying IP socket address.
  asio::detail::inet_addr_storage_type data_;
};

/// Output an endpoint as a string.
/**
 * Used to output a human-readable string for a specified endpoint.
 *
 * @param os The output stream to which the string will be written.
 *
 * @param endpoint The endpoint to be written.
 *
 * @return The output stream.
 *
 * @relates asio::ipv6::basic_endpoint
 */
#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
template <typename Protocol>
std::ostream& operator<<(std::ostream& os,
    const basic_endpoint<Protocol>& endpoint)
{
  const address& addr = endpoint.address();
  if (addr.is_ipv4())
    os << addr.to_string();
  else
    os << '[' << addr.to_string() << ']';
  os << ':' << endpoint.port();
  return os;
}
#else // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
template <typename Ostream, typename Protocol>
Ostream& operator<<(Ostream& os, const basic_endpoint<Protocol>& endpoint)
{
  const address& addr = endpoint.address();
  if (addr.is_ipv4())
    os << addr.to_string();
  else
    os << '[' << addr.to_string() << ']';
  os << ':' << endpoint.port();
  return os;
}
#endif // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))

} // namespace ipv6
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_BASIC_ENDPOINT_HPP
