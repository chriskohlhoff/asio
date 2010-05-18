//
// ip/impl/address_v6.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_IMPL_ADDRESS_V6_HPP
#define ASIO_IP_IMPL_ADDRESS_V6_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/ip/address_v6.hpp"

#if !defined(BOOST_NO_IOSTREAM)
# include "asio/detail/push_options.hpp"
# include <ios>
# include "asio/detail/pop_options.hpp"
# include "asio/detail/throw_error.hpp"
#endif // !defined(BOOST_NO_IOSTREAM)

namespace asio {
namespace ip {

inline unsigned long address_v6::scope_id() const
{
  return scope_id_;
}

inline void address_v6::scope_id(unsigned long id)
{
  scope_id_ = id;
}

inline bool operator!=(const address_v6& a1, const address_v6& a2)
{
  return !(a1 == a2);
}

inline bool operator>(const address_v6& a1, const address_v6& a2)
{
  return a2 < a1;
}

inline bool operator<=(const address_v6& a1, const address_v6& a2)
{
  return !(a2 < a1);
}

inline bool operator>=(const address_v6& a1, const address_v6& a2)
{
  return !(a1 < a2);
}

inline address_v6 address_v6::any()
{
  return address_v6();
}

#if !defined(BOOST_NO_IOSTREAM)

template <typename Elem, typename Traits>
std::basic_ostream<Elem, Traits>& operator<<(
    std::basic_ostream<Elem, Traits>& os, const address_v6& addr)
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

#endif // ASIO_IP_IMPL_ADDRESS_V6_HPP
