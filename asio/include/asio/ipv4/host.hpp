//
// host.hpp
// ~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_HOST_HPP
#define ASIO_IPV4_HOST_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <string>
#include <vector>
#include "asio/detail/pop_options.hpp"

#include "asio/ipv4/address.hpp"

namespace asio {
namespace ipv4 {

/// Encapsulates information about an IP version 4 host.
/**
 * The asio::ipv4::host structure contains properties which describe a host.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
struct host
{
  /// The name of the host.
  std::string name;

  /// Alternate names for the host.
  std::vector<std::string> aliases;

  /// All IP addresses of the host.
  std::vector<address> addresses;
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_HOST_HPP
