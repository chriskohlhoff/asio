//
// ip/impl/host_name.ipp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2017 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_IMPL_HOST_NAME_IPP
#define ASIO_IP_IMPL_HOST_NAME_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/winsock_init.hpp"
#include "asio/ip/host_name.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

ns_string host_name()
{
  asio::detail::ns_char_t name[256];
  asio::error_code ec;
  if (asio::detail::socket_ops::gethostname(name, 
    sizeof(name) / sizeof(asio::detail::ns_char_t), ec) != 0)
  {
    asio::detail::throw_error(ec);
    return ns_string();
  }
  return ns_string(name);
}

ns_string host_name(asio::error_code& ec)
{
  asio::detail::ns_char_t name[256];
  if (asio::detail::socket_ops::gethostname(name, 
    sizeof(name) / sizeof(asio::detail::ns_char_t), ec) != 0)
    return ns_string();
  return ns_string(name);
}

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_IMPL_HOST_NAME_IPP
