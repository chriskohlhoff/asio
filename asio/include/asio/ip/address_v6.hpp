//
// address_v6.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_ADDRESS_V6_HPP
#define ASIO_IP_ADDRESS_V6_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstring>
#include <string>
#include <stdexcept>
#include <boost/array.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/error_handler.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ip {

/// Implements IP version 6 style addresses.
/**
 * The asio::ip::address_v6 class provides the ability to use and
 * manipulate IP version 6 addresses.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
class address_v6
{
public:
  /// The type used to represent an address as an array of bytes.
  typedef boost::array<unsigned char, 16> bytes_type;

  /// Default constructor.
  address_v6()
    : scope_id_(0)
  {
    asio::detail::in6_addr_type tmp_addr = IN6ADDR_ANY_INIT;
    addr_ = tmp_addr;
  }

  /// Construct an address from raw bytes and scope ID.
  explicit address_v6(const bytes_type& bytes, unsigned long scope_id = 0)
    : scope_id_(scope_id)
  {
    using namespace std; // For memcpy.
    memcpy(addr_.s6_addr, bytes.elems, 16);
  }

  /// Copy constructor.
  address_v6(const address_v6& other)
    : addr_(other.addr_),
      scope_id_(other.scope_id_)
  {
  }

  /// Assign from another address.
  address_v6& operator=(const address_v6& other)
  {
    addr_ = other.addr_;
    scope_id_ = other.scope_id_;
    return *this;
  }

  /// Get the scope ID of the address.
  unsigned long scope_id() const
  {
    return scope_id_;
  }

  /// Set the scope ID of the address.
  void scope_id(unsigned long id)
  {
    scope_id_ = id;
  }

  /// Get the address in bytes.
  bytes_type to_bytes() const
  {
    using namespace std; // For memcpy.
    bytes_type bytes;
    memcpy(bytes.elems, addr_.s6_addr, 16);
    return bytes;
  }

  /// Get the address as a string.
  std::string to_string() const
  {
    return to_string(asio::throw_error());
  }

  /// Get the address as a string.
  template <typename Error_Handler>
  std::string to_string(Error_Handler error_handler) const
  {
    char addr_str[asio::detail::max_addr_v6_str_len];
    const char* addr =
      asio::detail::socket_ops::inet_ntop(AF_INET6, &addr_, addr_str,
          asio::detail::max_addr_v6_str_len, scope_id_);
    if (addr == 0)
    {
      asio::error e(asio::detail::socket_ops::get_error());
      error_handler(e);
      return std::string();
    }
    asio::error e;
    error_handler(e);
    return addr;
  }

  /// Create an address from an IP address string.
  static address_v6 from_string(const char* str)
  {
    return from_string(str, asio::throw_error());
  }

  /// Create an address from an IP address string.
  template <typename Error_Handler>
  static address_v6 from_string(const char* str, Error_Handler error_handler)
  {
    address_v6 tmp;
    if (asio::detail::socket_ops::inet_pton(
          AF_INET6, str, &tmp.addr_, &tmp.scope_id_) <= 0)
    {
      asio::error e(asio::detail::socket_ops::get_error());
      error_handler(e);
      return address_v6();
    }
    asio::error e;
    error_handler(e);
    return tmp;
  }

  /// Create an address from an IP address string.
  static address_v6 from_string(const std::string& str)
  {
    return from_string(str.c_str(), asio::throw_error());
  }

  /// Create an address from an IP address string.
  template <typename Error_Handler>
  static address_v6 from_string(const std::string& str,
      Error_Handler error_handler)
  {
    return from_string(str.c_str(), error_handler);
  }

  /// Determine whether the address is a loopback address.
  bool is_loopback() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_LOOPBACK(&addr_) != 0;
  }

  /// Determine whether the address is unspecified.
  bool is_unspecified() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_UNSPECIFIED(&addr_) != 0;
  }

  /// Determine whether the address is link local.
  bool is_link_local() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_LINKLOCAL(&addr_) != 0;
  }

  /// Determine whether the address is site local.
  bool is_site_local() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_SITELOCAL(&addr_) != 0;
  }

  /// Determine whether the address is a mapped IPv4 address.
  bool is_ipv4_mapped() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_V4MAPPED(&addr_) != 0;
  }

  /// Determine whether the address is an IPv4-compatible address.
  bool is_ipv4_compatible() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_V4COMPAT(&addr_) != 0;
  }

  /// Determine whether the address is a multicast address.
  bool is_multicast() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_MULTICAST(&addr_) != 0;
  }

  /// Determine whether the address is a global multicast address.
  bool is_multicast_global() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_MC_GLOBAL(&addr_) != 0;
  }

  /// Determine whether the address is a link-local multicast address.
  bool is_multicast_link_local() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_MC_LINKLOCAL(&addr_) != 0;
  }

  /// Determine whether the address is a node-local multicast address.
  bool is_multicast_node_local() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_MC_NODELOCAL(&addr_) != 0;
  }

  /// Determine whether the address is a org-local multicast address.
  bool is_multicast_org_local() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_MC_ORGLOCAL(&addr_) != 0;
  }

  /// Determine whether the address is a site-local multicast address.
  bool is_multicast_site_local() const
  {
    using namespace asio::detail;
    return IN6_IS_ADDR_MC_SITELOCAL(&addr_) != 0;
  }

  /// Compare two addresses for equality.
  friend bool operator==(const address_v6& a1, const address_v6& a2)
  {
    using namespace std; // For memcmp.
    return memcmp(&a1.addr_, &a2.addr_,
        sizeof(asio::detail::in6_addr_type)) == 0
      && a1.scope_id_ == a2.scope_id_;
  }

  /// Compare two addresses for inequality.
  friend bool operator!=(const address_v6& a1, const address_v6& a2)
  {
    using namespace std; // For memcmp.
    return memcmp(&a1.addr_, &a2.addr_,
        sizeof(asio::detail::in6_addr_type)) != 0
      || a1.scope_id_ != a2.scope_id_;
  }

  /// Compare addresses for ordering.
  friend bool operator<(const address_v6& a1, const address_v6& a2)
  {
    using namespace std; // For memcmp.
    int memcmp_result = memcmp(&a1.addr_, &a2.addr_,
        sizeof(asio::detail::in6_addr_type)) < 0;
    if (memcmp_result < 0)
      return true;
    if (memcmp_result > 0)
      return false;
    return a1.scope_id_ < a2.scope_id_;
  }

  /// Obtain an address object that represents any address.
  static address_v6 any()
  {
    return address_v6();
  }

  /// Obtain an address object that represents the loopback address.
  static address_v6 loopback()
  {
    address_v6 tmp;
    asio::detail::in6_addr_type tmp_addr = IN6ADDR_LOOPBACK_INIT;
    tmp.addr_ = tmp_addr;
    return tmp;
  }

private:
  // The underlying IPv6 address.
  asio::detail::in6_addr_type addr_;

  // The scope ID associated with the address.
  unsigned long scope_id_;
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
 * @relates asio::ip::address_v6
 */
template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
    std::basic_ostream<Elem, Traits>& os, const address_v6& addr)
{
  os << addr.to_string();
  return os;
}

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_ADDRESS_V6_HPP
