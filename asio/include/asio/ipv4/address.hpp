//
// address.hpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_ADDRESS_HPP
#define ASIO_IPV4_ADDRESS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <string>
#include <boost/array.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/error_handler.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv4 {

/// Implements IP version 4 style addresses.
/**
 * The asio::ipv4::address class provides the ability to use and
 * manipulate IP version 4 addresses.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
class address
{
public:
  /// The type used to represent an address as an array of bytes.
  typedef boost::array<unsigned char, 4> bytes_type;

  /// Default constructor.
  address()
  {
    addr_.s_addr = 0;
  }

  /// Construct an address from raw bytes.
  address(const bytes_type& bytes)
  {
    using namespace std; // For memcpy.
    memcpy(&addr_.s_addr, bytes.elems, 4);
  }

  /// Construct an address from a unsigned long in host byte order.
  address(unsigned long addr)
  {
    addr_.s_addr = asio::detail::socket_ops::host_to_network_long(addr);
  }

  /// Construct an address using an IP address string in dotted decimal form.
  address(const char* host)
  {
    if (asio::detail::socket_ops::inet_pton(AF_INET, host, &addr_) <= 0)
    {
      asio::error e(asio::detail::socket_ops::get_error());
      boost::throw_exception(e);
    }
  }

  /// Construct an address using an IP address string in dotted decimal form.
  template <typename Error_Handler>
  address(const char* host, Error_Handler error_handler)
  {
    if (asio::detail::socket_ops::inet_pton(AF_INET, host, &addr_) <= 0)
    {
      asio::error e(asio::detail::socket_ops::get_error());
      error_handler(e);
    }
  }

  /// Construct an address using an IP address string in dotted decimal form.
  address(const std::string& host)
  {
    if (asio::detail::socket_ops::inet_pton(
          AF_INET, host.c_str(), &addr_) <= 0)
    {
      asio::error e(asio::detail::socket_ops::get_error());
      boost::throw_exception(e);
    }
  }

  /// Construct an address using an IP address string in dotted decimal form.
  template <typename Error_Handler>
  address(const std::string& host, Error_Handler error_handler)
  {
    if (asio::detail::socket_ops::inet_pton(
          AF_INET, host.c_str(), &addr_) <= 0)
    {
      asio::error e(asio::detail::socket_ops::get_error());
      error_handler(e);
    }
  }

  /// Copy constructor.
  address(const address& other)
    : addr_(other.addr_)
  {
  }

  /// Assign from another address.
  address& operator=(const address& other)
  {
    addr_ = other.addr_;
    return *this;
  }

  /// Assign from an unsigned long.
  address& operator=(unsigned long addr)
  {
    addr_.s_addr = asio::detail::socket_ops::host_to_network_long(addr);
    return *this;
  }

  /// Assign from an IP address string in dotted decimal form.
  address& operator=(const char* addr)
  {
    address tmp(addr);
    addr_ = tmp.addr_;
    return *this;
  }

  /// Assign from an IP address string in dotted decimal form.
  address& operator=(const std::string& addr)
  {
    address tmp(addr);
    addr_ = tmp.addr_;
    return *this;
  }

  /// Get the address in bytes.
  bytes_type to_bytes() const
  {
    using namespace std; // For memcpy.
    bytes_type bytes;
    memcpy(bytes.elems, &addr_.s_addr, 4);
    return bytes;
  }

  /// Get the address as an unsigned long in host byte order
  unsigned long to_ulong() const
  {
    return asio::detail::socket_ops::network_to_host_long(addr_.s_addr);
  }

  /// Get the address as a string in dotted decimal format.
  std::string to_string() const
  {
    return to_string(asio::throw_error());
  }

  /// Get the address as a string in dotted decimal format.
  template <typename Error_Handler>
  std::string to_string(Error_Handler error_handler) const
  {
    char addr_str[asio::detail::max_addr_v4_str_len];
    const char* addr =
      asio::detail::socket_ops::inet_ntop(AF_INET, &addr_, addr_str,
          asio::detail::max_addr_v4_str_len);
    if (addr == 0)
    {
      asio::error e(asio::detail::socket_ops::get_error());
      error_handler(e);
      return std::string();
    }
    return addr;
  }

  /// Determine whether the address is a class A address.
  bool is_class_A() const
  {
    return IN_CLASSA(to_ulong());
  }

  /// Determine whether the address is a class B address.
  bool is_class_B() const
  {
    return IN_CLASSB(to_ulong());
  }

  /// Determine whether the address is a class C address.
  bool is_class_C() const
  {
    return IN_CLASSC(to_ulong());
  }

  /// Determine whether the address is a class D address.
  bool is_class_D() const
  {
    return IN_CLASSD(to_ulong());
  }

  /// Determine whether the address is a multicast address.
  bool is_multicast() const
  {
    return IN_MULTICAST(to_ulong());
  }

  /// Compare two addresses for equality.
  friend bool operator==(const address& a1, const address& a2)
  {
    return a1.addr_.s_addr == a2.addr_.s_addr;
  }

  /// Compare two addresses for inequality.
  friend bool operator!=(const address& a1, const address& a2)
  {
    return a1.addr_.s_addr != a2.addr_.s_addr;
  }

  /// Compare addresses for ordering.
  friend bool operator<(const address& a1, const address& a2)
  {
    return a1.to_ulong() < a2.to_ulong();
  }

  /// Obtain an address object that represents any address.
  static address any()
  {
    return address(static_cast<unsigned long>(INADDR_ANY));
  }

  /// Obtain an address object that represents the loopback address.
  static address loopback()
  {
    return address(static_cast<unsigned long>(INADDR_LOOPBACK));
  }

  /// Obtain an address object that represents the broadcast address.
  static address broadcast()
  {
    return address(static_cast<unsigned long>(INADDR_BROADCAST));
  }

private:
  // The underlying IPv4 address.
  in_addr addr_;
};

/// Output an address as a string.
/**
 * Used to output a human-readable string for a specified address.
 *
 * @param os The output stream to which the string will be written.
 *
 * @param addr The address to be written.
 *
 * @return The output stream.
 *
 * @relates tcp::endpoint
 */
template <typename Ostream>
Ostream& operator<<(Ostream& os, const address& addr)
{
  os << addr.to_string();
  return os;
}

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_ADDRESS_HPP
