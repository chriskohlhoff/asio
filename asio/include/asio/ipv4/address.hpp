//
// address.hpp
// ~~~~~~~~~~~
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

#ifndef ASIO_IPV4_ADDRESS_HPP
#define ASIO_IPV4_ADDRESS_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <string>
#include "asio/detail/pop_options.hpp"

#include "asio/socket_error.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv4 {

/// Implements IP version 4 style addresses.
/**
 * The asio::ipv4::address class provides the ability to use and manipulate IP
 * version 4 addresses.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
class address
{
public:
  /// Default constructor.
  address()
  {
    addr_.s_addr = 0;
  }

  /// Construct an address from a unsigned long in host byte order.
  address(unsigned long addr)
  {
    addr_.s_addr = htonl(addr);
  }

  /// Construct an address using an IP address string in dotted decimal form.
  address(const std::string& host)
  {
    if (asio::detail::socket_ops::inet_pton(AF_INET, host.c_str(), &addr_) <= 0)
      throw socket_error(asio::detail::socket_ops::get_error());
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
    addr_.s_addr = htonl(addr);
    return *this;
  }

  /// Assign from an IP address string in dotted decimal form.
  address& operator=(const std::string& addr)
  {
    address tmp(addr);
    addr_ = tmp.addr_;
    return *this;
  }

  /// Get the address as an unsigned long in host byte order
  unsigned long to_ulong() const
  {
    return ntohl(addr_.s_addr);
  }

  /// Get the address as a string in dotted decimal format.
  std::string to_string() const
  {
    char addr_str[asio::detail::max_addr_str_len];
    const char* addr = asio::detail::socket_ops::inet_ntop(AF_INET, &addr_,
        addr_str, asio::detail::max_addr_str_len);
    if (addr == 0)
      throw socket_error(asio::detail::socket_ops::get_error());
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
    return a1.addr_.s_addr < a2.addr_.s_addr;
  }

  /// Obtain an address object that represents any address.
  static address any()
  {
    return address(INADDR_ANY);
  }

  /// Obtain an address object that represents the loopback address.
  static address loopback()
  {
    return address(INADDR_LOOPBACK);
  }

  /// Obtain an address object that represents the broadcast address.
  static address broadcast()
  {
    return address(INADDR_BROADCAST);
  }

private:
  // The underlying IPv4 address.
  in_addr addr_;
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_ADDRESS_HPP
