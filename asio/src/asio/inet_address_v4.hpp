//
// inet_address_v4.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_INET_ADDRESS_V4_HPP
#define ASIO_INET_ADDRESS_V4_HPP

#include <string>
#include <boost/integer.hpp>
#include "asio/socket_address.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// The inet_address_v4 class implements IP version 4 style addresses.
class inet_address_v4
  : public socket_address
{
public:
  /// Underlying types for internet addresses.
  typedef boost::uint_t<16>::least port_type;
  typedef boost::uint_t<32>::least addr_type;

  /// Default constructor.
  inet_address_v4();

  /// Construct an address using a port number, specified in the host's byte
  /// order. The IP address will be the any address (i.e. INADDR_ANY). This
  /// constructor would typically be used for accepting new connections.
  inet_address_v4(port_type port_num);

  /// Construct an address using a port number and an IP address. This
  /// constructor may be used for accepting connections on a specific interface
  /// or for making a connection to a remote address.
  inet_address_v4(port_type port_num, addr_type host_addr);

  /// Construct an address using a port number and an IP address in dotted
  /// decimal form or a host name. This constructor may be used for accepting
  /// connections on a specific interface or for making a connection to a
  /// remote address.
  inet_address_v4(port_type port_num, const std::string& host);

  /// Copy constructor.
  inet_address_v4(const inet_address_v4& other);

  /// Assign from another inet_address_v4.
  inet_address_v4& operator=(const inet_address_v4& other);

  /// Destructor.
  virtual ~inet_address_v4();

  /// The address is good.
  virtual bool good() const;

  /// The address is bad.
  virtual bool bad() const;

  /// The address family.
  virtual int family() const;

  /// Get the underlying address in the native type.
  virtual native_address_type* native_address();

  /// Get the underlying address in the native type.
  virtual const native_address_type* native_address() const;

  /// Get the underlying size of the address in the native type.
  virtual native_size_type native_size() const;

  /// Set the underlying size of the address in the native type.
  virtual void native_size(native_size_type size);

  /// Get the port associated with the address. The port number is always in
  /// the host's byte order.
  port_type port() const;

  /// Set the port associated with the address. The port number is always in
  /// the host's byte order.
  void port(port_type port_num);

  /// Get the host associated with the address in the numeric form.
  addr_type host_addr() const;

  /// Set the host associated with the address.
  void host_addr(addr_type host);

  /// Get the host's address in dotted decimal format.
  std::string host_addr_str() const;

  /// Set the host's address using dotted decimal format.
  void host_addr_str(const std::string& host);

  /// Get the host name.
  std::string host_name() const;

  /// Set the host name.
  void host_name(const std::string& host);

private:
  // The underlying IPv4 socket address.
  detail::inet_addr_v4_type addr_;

  // Whether the address is valid.
  bool good_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_INET_ADDRESS_V4_HPP
