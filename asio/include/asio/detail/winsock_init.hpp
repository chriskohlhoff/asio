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

  // Used to ensure that the winsock library is initialised.
  static void use()
  {
    while (&instance_ == 0);
  }

private:
  // Instance to force initialisation of winsock at global scope.
  static winsock_init<Major, Minor> instance_;
};

template <int Major, int Minor>
winsock_init<Major, Minor> winsock_init<Major, Minor>::instance_;

} // namespace detail
} // namespace asio

#endif // _WIN32

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WINSOCK_INIT_HPP
