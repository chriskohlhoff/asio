//
// host_resolver.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
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
