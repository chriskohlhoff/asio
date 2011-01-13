//
// ip/detail/impl/endpoint.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_DETAIL_IMPL_ENDPOINT_IPP
#define ASIO_IP_DETAIL_IMPL_ENDPOINT_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <cstring>
#if !defined(BOOST_NO_IOSTREAM)
# include <sstream>
#endif // !defined(BOOST_NO_IOSTREAM)
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"
#include "asio/ip/detail/endpoint.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {
namespace detail {

endpoint::endpoint()
  : data_()
{
  data_.v4.sin_family = AF_INET;
  data_.v4.sin_port = 0;
  data_.v4.sin_addr.s_addr = INADDR_ANY;
}

endpoint::endpoint(int family, unsigned short port_num)
  : data_()
{
  using namespace std; // For memcpy.
  if (family == PF_INET)
  {
    data_.v4.sin_family = AF_INET;
    data_.v4.sin_port =
      asio::detail::socket_ops::host_to_network_short(port_num);
    data_.v4.sin_addr.s_addr = INADDR_ANY;
  }
  else
  {
    data_.v6.sin6_family = AF_INET6;
    data_.v6.sin6_port =
      asio::detail::socket_ops::host_to_network_short(port_num);
    data_.v6.sin6_flowinfo = 0;
    asio::detail::in6_addr_type tmp_addr = IN6ADDR_ANY_INIT;
    data_.v6.sin6_addr = tmp_addr;
    data_.v6.sin6_scope_id = 0;
  }
}

endpoint::endpoint(const asio::ip::address& addr,
    unsigned short port_num)
  : data_()
{
  using namespace std; // For memcpy.
  if (addr.is_v4())
  {
    data_.v4.sin_family = AF_INET;
    data_.v4.sin_port =
      asio::detail::socket_ops::host_to_network_short(port_num);
    data_.v4.sin_addr.s_addr =
      asio::detail::socket_ops::host_to_network_long(
          addr.to_v4().to_ulong());
  }
  else
  {
    data_.v6.sin6_family = AF_INET6;
    data_.v6.sin6_port =
      asio::detail::socket_ops::host_to_network_short(port_num);
    data_.v6.sin6_flowinfo = 0;
    asio::ip::address_v6 v6_addr = addr.to_v6();
    asio::ip::address_v6::bytes_type bytes = v6_addr.to_bytes();
    memcpy(data_.v6.sin6_addr.s6_addr, bytes.elems, 16);
    data_.v6.sin6_scope_id = v6_addr.scope_id();
  }
}

void endpoint::resize(std::size_t size)
{
  if (size > sizeof(asio::detail::sockaddr_storage_type))
  {
    asio::error_code ec(asio::error::invalid_argument);
    asio::detail::throw_error(ec);
  }
}

unsigned short endpoint::port() const
{
  if (is_v4())
  {
    return asio::detail::socket_ops::network_to_host_short(
        data_.v4.sin_port);
  }
  else
  {
    return asio::detail::socket_ops::network_to_host_short(
        data_.v6.sin6_port);
  }
}

void endpoint::port(unsigned short port_num)
{
  if (is_v4())
  {
    data_.v4.sin_port
      = asio::detail::socket_ops::host_to_network_short(port_num);
  }
  else
  {
    data_.v6.sin6_port
      = asio::detail::socket_ops::host_to_network_short(port_num);
  }
}

asio::ip::address endpoint::address() const
{
  using namespace std; // For memcpy.
  if (is_v4())
  {
    return asio::ip::address_v4(
        asio::detail::socket_ops::network_to_host_long(
          data_.v4.sin_addr.s_addr));
  }
  else
  {
    asio::ip::address_v6::bytes_type bytes;
    memcpy(bytes.elems, data_.v6.sin6_addr.s6_addr, 16);
    return asio::ip::address_v6(bytes, data_.v6.sin6_scope_id);
  }
}

void endpoint::address(const asio::ip::address& addr)
{
  endpoint tmp_endpoint(addr, port());
  data_ = tmp_endpoint.data_;
}

bool operator==(const endpoint& e1, const endpoint& e2)
{
  return e1.address() == e2.address() && e1.port() == e2.port();
}

bool operator<(const endpoint& e1, const endpoint& e2)
{
  if (e1.address() < e2.address())
    return true;
  if (e1.address() != e2.address())
    return false;
  return e1.port() < e2.port();
}

#if !defined(BOOST_NO_IOSTREAM)
std::string endpoint::to_string(asio::error_code& ec) const
{
  std::string a = address().to_string(ec);
  if (ec)
    return std::string();

  std::ostringstream tmp_os;
  tmp_os.imbue(std::locale::classic());
  if (is_v4())
    tmp_os << a;
  else
    tmp_os << '[' << a << ']';
  tmp_os << ':' << port();

  return tmp_os.str();
}
#endif // !defined(BOOST_NO_IOSTREAM)

} // namespace detail
} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_DETAIL_IMPL_ENDPOINT_IPP
