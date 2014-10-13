//
// ip/impl/network_v6.ipp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// Copyright (c) 2014 Oliver Kowalke (oliver dot kowalke at gmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_IMPL_NETWORK_V6_IPP
#define ASIO_IP_IMPL_NETWORK_V6_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "asio/error.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/throw_exception.hpp"
#include "asio/ip/network_v6.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

network_v6::network_v6(const address_v6& addr, unsigned short prefix_len)
  : address_(addr),
    prefix_length_(prefix_len)
{
  if (prefix_len > 128)
  {
    std::out_of_range ex("prefix length too large");
    asio::detail::throw_exception(ex);
  }
}

network_v6::network_v6(const address_v6& addr, const address_v6& mask)
  : address_(addr),
    prefix_length_(0)
{
  address_v6::bytes_type mask_bytes = mask.to_bytes();
  bool finished = false;
  for (std::size_t i = 0; i < mask_bytes.size(); ++i)
  {
    if (finished)
    {
      if (mask_bytes[i])
      {
        std::invalid_argument ex("non-contiguous netmask");
        asio::detail::throw_exception(ex);
      }
      continue;
    }
    else
    {
      switch (mask_bytes[i])
      {
      case 255:
        prefix_length_ += 8;
        break;
      case 254: // prefix_length_ += 7
        prefix_length_ += 1;
      case 252: // prefix_length_ += 6
        prefix_length_ += 1;
      case 248: // prefix_length_ += 5
        prefix_length_ += 1;
      case 240: // prefix_length_ += 4
        prefix_length_ += 1;
      case 224: // prefix_length_ += 3
        prefix_length_ += 1;
      case 192: // prefix_length_ += 2
        prefix_length_ += 1;
      case 128: // prefix_length_ += 1
        prefix_length_ += 1;
      case 0:   // nbits += 0
        finished = true;
        break;
      default:
        std::out_of_range ex("non-contiguous netmask");
        asio::detail::throw_exception(ex);
      }
    }
  }
}

address_v6 network_v6::netmask() const ASIO_NOEXCEPT
{
  using namespace std; // for memset
  detail::in6_addr_type mask;
  memset(&mask,0,sizeof(mask));

  const int bc = (prefix_length_ / 32);
  const int rest = (prefix_length_ % 32);

  for ( int i = 0; i < bc; i++)
      mask.s6_addr32[i] = 0xffffffff;

  if ( rest)
      mask.s6_addr32[bc] = ntohl( ( ( uint32_t)0xffffffff) << ( 32 - rest) );

  address_v6::bytes_type bytes;
  memcpy(bytes.data(), mask.s6_addr, 16);
  return address_v6(bytes);
}

address_range_v6 network_v6::hosts() const ASIO_NOEXCEPT
{
  if ( is_host() ) {
    return address_range_v6(address_, address_.successor_());
  } else {
    address_v6 netw = network();
    return address_range_v6(network().successor_(), netw);
  }
}

bool network_v6::is_subnet_of(const network_v6& other) const
{
  if (other.prefix_length_ >= prefix_length_)
    return false; // Only real subsets are allowed.
  const network_v6 me(address_, other.prefix_length_);
  return other.canonical() == me.canonical();
}

std::string network_v6::to_string() const
{
  asio::error_code ec;
  std::string addr = to_string(ec);
  asio::detail::throw_error(ec);
  return addr;
}

std::string network_v6::to_string(asio::error_code& ec) const
{
  ec = asio::error_code();
  char prefix_len[16];
#if defined(ASIO_HAS_SECURE_RTL)
  sprintf_s(prefix_len, sizeof(prefix_len), "/%u", prefix_length_);
#else // defined(ASIO_HAS_SECURE_RTL)
  sprintf(prefix_len, "/%u", prefix_length_);
#endif // defined(ASIO_HAS_SECURE_RTL)
  return address_.to_string() + prefix_len;
}

network_v6 make_network_v6(const char* str)
{
  return make_network_v6(std::string(str));
}

network_v6 make_network_v6(const char* str, asio::error_code& ec)
{
  return make_network_v6(std::string(str), ec);
}

network_v6 make_network_v6(const std::string& str)
{
  asio::error_code ec;
  network_v6 net = make_network_v6(str, ec);
  asio::detail::throw_error(ec);
  return net;
}

network_v6 make_network_v6(const std::string& str,
    asio::error_code& ec)
{
  std::string::size_type pos = str.find_first_of("/");

  if (pos == std::string::npos)
  {
    ec = asio::error::invalid_argument;
    return network_v6();
  }

  if (pos == str.size() - 1)
  {
    ec = asio::error::invalid_argument;
    return network_v6();
  }

  std::string::size_type end = str.find_first_not_of("0123456789", pos+1);
  if (end != std::string::npos)
  {
    ec = asio::error::invalid_argument;
    return network_v6();
  }

  return network_v6(make_address_v6(str.substr(0, pos)),
      std::atoi(str.substr(pos + 1).c_str()));
}

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_IMPL_NETWORK_V6_IPP
