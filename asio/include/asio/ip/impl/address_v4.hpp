//
// ip/impl/address_v4.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_IMPL_ADDRESS_V4_HPP
#define ASIO_IP_IMPL_ADDRESS_V4_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/ip/address_v4.hpp"

#if !defined(BOOST_NO_IOSTREAM)
# include "asio/detail/push_options.hpp"
# include <ios>
# include "asio/detail/pop_options.hpp"
# include "asio/detail/throw_error.hpp"
#endif // !defined(BOOST_NO_IOSTREAM)

namespace asio {
namespace ip {

inline address_v4::address_v4()
{
  addr_.s_addr = 0;
}

inline address_v4::address_v4(const address_v4& other)
  : addr_(other.addr_)
{
}

inline address_v4& address_v4::operator=(const address_v4& other)
{
  addr_ = other.addr_;
  return *this;
}

inline bool operator==(const address_v4& a1, const address_v4& a2)
{
  return a1.addr_.s_addr == a2.addr_.s_addr;
}

inline bool operator!=(const address_v4& a1, const address_v4& a2)
{
  return a1.addr_.s_addr != a2.addr_.s_addr;
}

inline bool operator<(const address_v4& a1, const address_v4& a2)
{
  return a1.to_ulong() < a2.to_ulong();
}

inline bool operator>(const address_v4& a1, const address_v4& a2)
{
  return a1.to_ulong() > a2.to_ulong();
}

inline bool operator<=(const address_v4& a1, const address_v4& a2)
{
  return a1.to_ulong() <= a2.to_ulong();
}

inline bool operator>=(const address_v4& a1, const address_v4& a2)
{
  return a1.to_ulong() >= a2.to_ulong();
}

inline address_v4 address_v4::any()
{
  return address_v4(static_cast<unsigned long>(INADDR_ANY));
}

inline address_v4 address_v4::loopback()
{
  return address_v4(static_cast<unsigned long>(INADDR_LOOPBACK));
}

inline address_v4 address_v4::broadcast()
{
  return address_v4(static_cast<unsigned long>(INADDR_BROADCAST));
}

#if !defined(BOOST_NO_IOSTREAM)

template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
    std::basic_ostream<Elem, Traits>& os, const address_v4& addr)
{
  asio::error_code ec;
  std::string s = addr.to_string(ec);
  if (ec)
  {
    if (os.exceptions() & std::ios::failbit)
      asio::detail::throw_error(ec);
    else
      os.setstate(std::ios_base::failbit);
  }
  else
    for (std::string::iterator i = s.begin(); i != s.end(); ++i)
      os << os.widen(*i);
  return os;
}

#endif // !defined(BOOST_NO_IOSTREAM)

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_IMPL_ADDRESS_V4_HPP
