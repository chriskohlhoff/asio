//
// generic_address.hpp
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

#ifndef ASIO_GENERIC_ADDRESS_HPP
#define ASIO_GENERIC_ADDRESS_HPP

#include "asio/detail/push_options.hpp"

#include "asio/socket_address.hpp"

namespace asio {

/// The generic_address class may be used to hold any type of socket address.
class generic_address
  : public socket_address
{
public:
  /// Default constructor.
  generic_address();

  /// Copy constructor.
  generic_address(const generic_address& other);

  /// Construct a copy of another socket_address.
  generic_address(const socket_address& other);

  /// Assign from another generic address.
  generic_address& operator=(const generic_address& other);

  /// Assign from another socket_address.
  generic_address& operator=(const socket_address& other);

  /// Destructor.
  virtual ~generic_address();

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

private:
  // Buffer used to hold the socket address.
  struct
  {
    native_address_type addr;
    char padding[128];
  } addr_buf_;

  // The size of the socket address buffer.
  native_size_type addr_size_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_GENERIC_ADDRESS_HPP
