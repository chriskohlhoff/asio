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

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
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
 *
 * @par Concepts:
 * CopyConstructible, Assignable.
 */
class host
{
public:
  /// Default constructor.
  host()
  {
  }

  /// Construct from component properties.
  host(const std::string& name, const asio::ipv4::address& addr)
    : name_(name)
  {
    addresses_.push_back(asio::ipv4::address::any());
  }

  /// Construct from component properties.
  template <typename Name_Iterator, typename Address_Iterator>
  host(const std::string& name, const asio::ipv4::address& addr,
      Name_Iterator alternate_names_begin,
      Name_Iterator alternate_names_end,
      Address_Iterator other_addresses_begin,
      Address_Iterator other_addresses_end)
    : name_(name)
  {
    addresses_.push_back(addr);
    alternate_names_.insert(alternate_names_.end(),
        alternate_names_begin, alternate_names_end);
    addresses_.insert(addresses_.end(),
        other_addresses_begin, other_addresses_end);
  }

  /// Get the name of the host.
  std::string name() const
  {
    return name_;
  }

  /// Get the number of alternate names for the host.
  std::size_t alternate_name_count() const
  {
    return alternate_names_.size();
  }

  /// Get the alternate name at the specified index.
  std::string alternate_name(std::size_t index) const
  {
    return alternate_names_[index];
  }

  /// Get the number of addresses for the host.
  std::size_t address_count() const
  {
    return addresses_.size();
  }

  /// Get the address at the specified index.
  asio::ipv4::address address(std::size_t index) const
  {
    return addresses_[index];
  }

  /// Swap the host object's values with another host.
  void swap(host& other)
  {
    name_.swap(other.name_);
    alternate_names_.swap(other.alternate_names_);
    addresses_.swap(other.addresses_);
  }

private:
  // The name of the host.
  std::string name_;

  // Alternate names for the host.
  std::vector<std::string> alternate_names_;

  // All IP addresses of the host.
  std::vector<asio::ipv4::address> addresses_;
};

} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_HOST_HPP
