//
// winsock_init.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WINSOCK_INIT_HPP
#define ASIO_DETAIL_WINSOCK_INIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#if defined(_WIN32)

#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <int Major = 2, int Minor = 0>
class winsock_init
{
public:
  // Constructor.
  winsock_init()
  {
    WSADATA wsa_data;
    ::WSAStartup(MAKEWORD(Major, Minor), &wsa_data);
  }

  // Destructor.
  ~winsock_init()
  {
    ::WSACleanup();
  }
};

} // namespace detail
} // namespace asio

#endif // _WIN32

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WINSOCK_INIT_HPP
