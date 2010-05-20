//
// impl/error.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IMPL_ERROR_HPP
#define ASIO_IMPL_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace error {

inline asio::error_code make_error_code(basic_errors e)
{
  return asio::error_code(
      static_cast<int>(e), get_system_category());
}

inline asio::error_code make_error_code(netdb_errors e)
{
  return asio::error_code(
      static_cast<int>(e), get_netdb_category());
}

inline asio::error_code make_error_code(addrinfo_errors e)
{
  return asio::error_code(
      static_cast<int>(e), get_addrinfo_category());
}

inline asio::error_code make_error_code(misc_errors e)
{
  return asio::error_code(
      static_cast<int>(e), get_misc_category());
}

inline asio::error_code make_error_code(ssl_errors e)
{
  return asio::error_code(
      static_cast<int>(e), get_ssl_category());
}

} // namespace error
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IMPL_ERROR_HPP
