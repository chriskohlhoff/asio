//
// host.hpp
// ~~~~~~~~
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
