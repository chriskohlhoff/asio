//
// winsock_init.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
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
