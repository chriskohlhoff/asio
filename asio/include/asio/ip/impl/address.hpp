//
// ip/impl/address.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_IMPL_ADDRESS_HPP
#define ASIO_IP_IMPL_ADDRESS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/ip/address.hpp"

namespace asio {
namespace ip {

inline bool address::is_v4() const
{
  return type_ == ipv4;
}

inline bool address::is_v6() const
{
  return type_ == ipv6;
}

inline bool operator!=(const address& a1, const address& a2)
{
  return !(a1 == a2);
}

inline bool operator>(const address& a1, const address& a2)
{
  return a2 < a1;
}

inline bool operator<=(const address& a1, const address& a2)
{
  return !(a2 < a1);
}

inline bool operator>=(const address& a1, const address& a2)
{
  return !(a1 < a2);
}

#if !defined(BOOST_NO_IOSTREAM)

template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
    std::basic_ostream<Elem, Traits>& os, const address& addr)
{
  os << addr.to_string();
  return os;
}

#endif // !defined(BOOST_NO_IOSTREAM)

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_IMPL_ADDRESS_HPP
