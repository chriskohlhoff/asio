//
// generic_address.cpp
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

#include "asio/generic_address.hpp"

#include "asio/detail/push_options.hpp"
#include <string.h>
#include "asio/detail/pop_options.hpp"

namespace asio {

generic_address::
generic_address()
  : addr_buf_(),
    addr_size_(sizeof(addr_buf_))
{
}

generic_address::
generic_address(
    const generic_address& other)
  : addr_buf_(other.addr_buf_),
    addr_size_(other.addr_size_)
{
}

generic_address::
generic_address(
    const socket_address& other)
  : addr_buf_(),
    addr_size_(other.native_size())
{
  if (addr_size_ <= sizeof(addr_buf_))
    memcpy(&addr_buf_, other.native_address(), addr_size_);
}

generic_address&
generic_address::
operator=(
    const generic_address& other)
{
  addr_buf_ = other.addr_buf_;
  addr_size_ = other.addr_size_;
  return *this;
}

generic_address&
generic_address::
operator=(
    const socket_address& other)
{
  addr_size_ = other.native_size();;
  if (addr_size_ <= sizeof(addr_buf_))
    memcpy(&addr_buf_, other.native_address(), addr_size_);
  return *this;
}

generic_address::
~generic_address()
{
}

bool
generic_address::
good() const
{
  return addr_size_ <= sizeof(addr_buf_);
}

bool
generic_address::
bad() const
{
  return addr_size_ > sizeof(addr_buf_);
}

int
generic_address::
family() const
{
  return native_address()->sa_family;
}

socket_address::native_address_type*
generic_address::
native_address()
{
  return reinterpret_cast<socket_address::native_address_type*>(&addr_buf_);
}

const socket_address::native_address_type*
generic_address::
native_address() const
{
  return reinterpret_cast<const socket_address::native_address_type*>(
      &addr_buf_);
}

socket_address::native_size_type
generic_address::
native_size() const
{
  return addr_size_;
}

void
generic_address::
native_size(
    native_size_type size)
{
  addr_size_ = size;
}

} // namespace asio
