//
// socket_address.hpp
// ~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_SOCKET_ADDRESS_HPP
#define ASIO_SOCKET_ADDRESS_HPP

#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// The socket_address class is the base class for all supported types of
/// socket address structures.
class socket_address
{
public:
  /// The native types of the socket address. These types are dependent on the
  /// underlying implementation of the socket layer.
  typedef detail::socket_addr_type native_address_type;
  typedef detail::socket_addr_len_type native_size_type;

  /// Destructor.
  virtual ~socket_address();

  /// The address is good.
  virtual bool good() const = 0;

  /// The address is bad.
  virtual bool bad() const = 0;

  /// The address family.
  virtual int family() const = 0;

  /// Get the underlying address in the native type.
  virtual native_address_type* native_address() = 0;

  /// Get the underlying address in the native type.
  virtual const native_address_type* native_address() const = 0;

  /// Get the underlying size of the address in the native type.
  virtual native_size_type native_size() const = 0;

  /// Set the underlying size of the address in the native type.
  virtual void native_size(native_size_type size) = 0;

private:
  /// Prevent assignment directly to the base class.
  void operator=(const socket_address&);
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_ADDRESS_HPP
