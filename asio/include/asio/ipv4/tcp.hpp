//
// tcp.hpp
// ~~~~~~~
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

#ifndef ASIO_IPV4_TCP_HPP
#define ASIO_IPV4_TCP_HPP

#include "asio/detail/push_options.hpp"

#include "asio/socket_option.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ipv4 {

/// The udp class contains the flags necessary to use UDP.
class tcp
{
public:
  int type() const
  {
    return SOCK_STREAM;
  }

  int protocol() const
  {
    return IPPROTO_TCP;
  }

  int family() const
  {
    return PF_INET;
  }

  /// Socket option for disabling the Nagle algorithm.
  typedef socket_option::flag<IPPROTO_TCP, TCP_NODELAY> tcp_no_delay;
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_TCP_HPP
