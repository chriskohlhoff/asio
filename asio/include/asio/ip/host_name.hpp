//
// host_name.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_HOST_NAME_HPP
#define ASIO_IP_HOST_NAME_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <string>
#include "asio/detail/pop_options.hpp"

#include "asio/error_handler.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {
namespace ip {

/// Get the current host name.
std::string host_name();

/// Get the current host name.
template <typename Error_Handler>
std::string host_name(Error_Handler error_handler);

inline std::string host_name()
{
  return host_name(asio::throw_error());
}

template <typename Error_Handler>
std::string host_name(Error_Handler error_handler)
{
  char name[1024];
  if (asio::detail::socket_ops::gethostname(name, sizeof(name)) != 0)
  {
    asio::error error(asio::detail::socket_ops::get_error());
    error_handler(error);
    return std::string();
  }
  return std::string(name);
}

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_HOST_NAME_HPP
