//
// host_resolver.hpp
// ~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_IPV4_HOST_RESOLVER_HPP
#define ASIO_IPV4_HOST_RESOLVER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/demuxer.hpp"
#include "asio/ipv4/basic_host_resolver.hpp"
#include "asio/ipv4/detail/host_resolver_service.hpp"

namespace asio {
namespace ipv4 {

/// Typedef for the typical usage of host_resolver.
#if defined(GENERATING_DOCUMENTATION)
typedef basic_host_resolver
  <
    implementation_defined
  > host_resolver;
#else
typedef basic_host_resolver
  <
    asio::ipv4::detail::host_resolver_service<demuxer>
  > host_resolver;
#endif

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_HOST_RESOLVER_HPP
