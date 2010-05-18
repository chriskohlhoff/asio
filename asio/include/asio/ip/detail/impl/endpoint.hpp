//
// ip/detail/impl/endpoint.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_DETAIL_IMPL_ENDPOINT_HPP
#define ASIO_IP_DETAIL_IMPL_ENDPOINT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/ip/detail/endpoint.hpp"

namespace asio {
namespace ip {
namespace detail {

inline endpoint::endpoint(const endpoint& other)
  : data_(other.data_)
{
}

inline endpoint& endpoint::operator=(const endpoint& other)
{
  data_ = other.data_;
  return *this;
}

inline asio::detail::socket_addr_type* endpoint::data()
{
  return &data_.base;
}

inline const asio::detail::socket_addr_type* endpoint::data() const
{
  return &data_.base;
}

inline std::size_t endpoint::size() const
{
  if (is_v4())
    return sizeof(asio::detail::sockaddr_in4_type);
  else
    return sizeof(asio::detail::sockaddr_in6_type);
}

inline std::size_t endpoint::capacity() const
{
  return sizeof(asio::detail::sockaddr_storage_type);
}

inline bool endpoint::is_v4() const
{
  return data_.base.sa_family == AF_INET;
}

} // namespace detail
} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_DETAIL_IMPL_ENDPOINT_HPP
