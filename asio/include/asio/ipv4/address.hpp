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
#include <cstring>
#include <boost/integer.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/socket_error.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/ipv4/tcp.hpp"
#include "asio/ipv4/udp.hpp"

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
 *
 * @par Concepts:
 * Address.
 */
class address
{
public:
  /// The default stream-based protocol associated with the address type.
  typedef tcp default_stream_protocol;

  /// The default datagram-based protocol associated with the address type.
  typedef udp default_dgram_protocol;

  /// The native type of the address structure. This type is dependent on the
  /// underlying implementation of the socket layer.
  typedef detail::socket_addr_type native_address_type;

  /// The native type for the size of the address structure. This type is
  /// dependent on the underlying implementation of the socket layer.
  typedef detail::socket_addr_len_type native_size_type;

  /// Underlying types for internet addresses.
  typedef boost::uint_t<16>::least port_type;
  typedef boost::uint_t<32>::least addr_type;

  /// Default constructor.
  address()
  {
    addr_.sin_family = AF_INET;
    addr_.sin_port = 0;
    addr_.sin_addr.s_addr = INADDR_ANY;
  }

  /// Construct an address using a port number, specified in the host's byte
  /// order. The IP address will be the any address (i.e. INADDR_ANY). This
  /// constructor would typically be used for accepting new connections.
  address(port_type port_num)
  {
    addr_.sin_family = AF_INET;
    addr_.sin_port = htons(port_num);
    addr_.sin_addr.s_addr = INADDR_ANY;
  }

  /// Construct an address using a port number and an IP address. This
  /// constructor may be used for accepting connections on a specific interface
  /// or for making a connection to a remote address.
  address(port_type port_num, addr_type host_addr)
  {
    addr_.sin_family = AF_INET;
    addr_.sin_port = htons(port_num);
    addr_.sin_addr.s_addr = host_addr;
  }

  /// Construct an address using a port number and an IP address in dotted
  /// decimal form or a host name. This constructor may be used for accepting
  /// connections on a specific interface or for making a connection to a
  /// remote address.
  address(port_type port_num, const std::string& host)
  {
    addr_.sin_family = AF_INET;
    addr_.sin_port = htons(port_num);
    host_name(host);
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

  /// The address family.
  int family() const
  {
    return AF_INET;
  }

  /// Get the underlying address in the native type.
  native_address_type* native_address()
  {
    return reinterpret_cast<address::native_address_type*>(&addr_);
  }

  /// Get the underlying address in the native type.
  const native_address_type* native_address() const
  {
    return reinterpret_cast<const address::native_address_type*>(
        &addr_);
  }

  /// Get the underlying size of the address in the native type.
  native_size_type native_size() const
  {
    return sizeof(addr_);
  }

  /// Set the underlying size of the address in the native type.
  void native_size(native_size_type size)
  {
    if (size != sizeof(addr_))
      throw socket_error(socket_error::invalid_argument);
  }

  /// Get the port associated with the address. The port number is always in
  /// the host's byte order.
  port_type port() const
  {
    return ntohs(addr_.sin_port);
  }

  /// Set the port associated with the address. The port number is always in
  /// the host's byte order.
  void port(port_type port_num)
  {
    addr_.sin_port = htons(port_num);
  }

  /// Get the host associated with the address in the numeric form.
  addr_type host_addr() const
  {
    return addr_.sin_addr.s_addr;
  }

  /// Set the host associated with the address.
  void host_addr(addr_type host)
  {
    addr_.sin_addr.s_addr = host;
  }

  /// Get the host's address in dotted decimal format.
  std::string host_addr_str() const
  {
    char addr_str[detail::max_addr_str_len];
    const char* addr = detail::socket_ops::inet_ntop(AF_INET, &addr_.sin_addr,
        addr_str, detail::max_addr_str_len);
    if (addr == 0)
      throw socket_error(detail::socket_ops::get_error());
    return addr;
  }

  /// Set the host's address using dotted decimal format.
  void host_addr_str(const std::string& host)
  {
    if (detail::socket_ops::inet_pton(AF_INET, host.c_str(),
          &addr_.sin_addr) <= 0)
      throw socket_error(detail::socket_ops::get_error());
  }

  /// Get the host name.
  std::string host_name() const
  {
    hostent ent;
    char buf[8192] = ""; // Size recommended by Stevens, UNPv1.
    int error = 0;
    if (detail::socket_ops::gethostbyaddr_r(
          reinterpret_cast<const char*>(&addr_.sin_addr),
          sizeof(addr_.sin_addr), AF_INET, &ent, buf, sizeof(buf),
          &error) == 0)
      throw socket_error(error);
    return ent.h_name;
  }

  /// Set the host name.
  void host_name(const std::string& host)
  {
    using namespace std; // For memcpy.
    hostent ent;
    char buf[8192] = ""; // Size recommended by Stevens, UNPv1.
    int error = 0;
    if (detail::socket_ops::gethostbyname_r(host.c_str(), &ent, buf,
          sizeof(buf), &error) == 0)
      throw socket_error(error);
    memcpy(&addr_.sin_addr, ent.h_addr, sizeof(addr_.sin_addr));
  }

private:
  // The underlying IPv4 socket address.
  detail::inet_addr_v4_type addr_;
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_ADDRESS_HPP
