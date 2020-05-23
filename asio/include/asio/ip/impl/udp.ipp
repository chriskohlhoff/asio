//
// ip/impl/udp.ipp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_IMPL_UDP_IPP
#define ASIO_IP_IMPL_UDP_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/ip/udp.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)
asio::detail::apple_nw_ptr<nw_parameters_t>
udp::apple_nw_create_parameters() const
{
  asio::detail::apple_nw_ptr<nw_parameters_t> parameters(
      nw_parameters_create_secure_udp(
        NW_PARAMETERS_DISABLE_PROTOCOL,
        NW_PARAMETERS_DEFAULT_CONFIGURATION));

  asio::detail::apple_nw_ptr<nw_protocol_stack_t> protocol_stack(
      nw_parameters_copy_default_protocol_stack(parameters));

  asio::detail::apple_nw_ptr<nw_protocol_options_t> ip_options(
      nw_protocol_stack_copy_internet_protocol(protocol_stack));

  if (family_ == ASIO_OS_DEF(AF_INET))
    nw_ip_options_set_version(ip_options, nw_ip_version_4);
  else
    nw_ip_options_set_version(ip_options, nw_ip_version_6);

  return parameters;
}
#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_IMPL_UDP_IPP
