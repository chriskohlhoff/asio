//
// ip/cidr_address_v4.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//                    Oliver Kowalke (oliver dot kowalke at gmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_CIDR_ADDRESS_V4_HPP
#define ASIO_IP_CIDR_ADDRESS_V4_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include <cstddef>
#include <iterator>
#include <string>

#include "asio/ip/address_v4.hpp"
#include "asio/detail/config.hpp"
#include "asio/error_code.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

class address_iterator_v4
{
public:
  typedef std::ptrdiff_t difference_type;
  typedef address_v4 value_type;
  typedef const address_v4* pointer;
  typedef const address_v4& reference;
  typedef std::bidirectional_iterator_tag iterator_category;

  explicit address_iterator_v4(const address_v4& addr);

  const address_v4& operator*() const;
  const address_v4* operator->() const;

  address_iterator_v4& operator++();
  address_iterator_v4 operator++(int);

  address_iterator_v4& operator--();
  address_iterator_v4 operator--(int);

  friend bool operator==(const address_iterator_v4& a, const address_iterator_v4& b);
  friend bool operator!=(const address_iterator_v4& a, const address_iterator_v4& b);

private:
  address_v4 address_;
};


/// Implements IP version 4 style addresses in CIDR notation.
/**
 * The asio::ip::cidr_address_v4 class provides the ability to use and
 * manipulate IP version 4 addresses in CIDR notation.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 */
class cidr_address_v4 {
public:
    /// Default constructor.
    cidr_address_v4();
    
    /// Construct an CIDR address based on the passed address and prefix length
    ASIO_DECL cidr_address_v4(const address_v4& addr, std::size_t prefix_length);
    
    /// Construct an CIDR address based on the passed address and netmask
    ASIO_DECL cidr_address_v4(const address_v4& addr, const address_v4& mask);
    
    /// Copy constructor.
    cidr_address_v4(const cidr_address_v4& other);

#if defined(ASIO_HAS_MOVE)
    /// Move constructor.
    cidr_address_v4(cidr_address_v4&& other) {}
#endif // defined(ASIO_HAS_MOVE)

    /// Assign from another CIDR address.
    cidr_address_v4& operator=(const cidr_address_v4& other)
    {
      return *this;
    }

#if defined(ASIO_HAS_MOVE)
    /// Move-assign from another CIDR address.
    cidr_address_v4& operator=(cidr_address_v4&& other)
    {
      return *this;
    }
#endif // defined(ASIO_HAS_MOVE)

    /// Optain an address range object that represents the network range
    /// that corresponds to the specific network.
    ASIO_DECL address_v4 network() const;

    /// Optain an address object that represents the host address
    /// that corresponds to the specific host part.
    ASIO_DECL address_v4 host() const;
   
    /// Obtain the netmask that corresponds to the address, based on its
    /// CIDR address class.
    ASIO_DECL address_v4 netmask() const;

    /// Obtain an address object that represents the broadcast address that
    /// corresponds to the specified address and netmask.
    ASIO_DECL address_v4 broadcast() const;

    typedef address_iterator_v4 iterator;

    /// Optain an iterator poiting to the first address that belongs to
    /// the specific network segment.
    ASIO_DECL iterator begin() const;

    /// Optain an iterator pointing to the end of the address range beonging
    /// to the specifix network segment.
    ASIO_DECL iterator end() const;

    /// Optain an iterator as an result of searching the requested address
    /// in the specific network segment.
    ASIO_DECL iterator find(const address_v4& addr) const;
   
    /// Calculate CIDR prefix length from netmask.
    static std::size_t calculate_prefix_length(const address_v4& netmask);
    
    /// Calculate netmask from CIDR prefix length.
    static address_v4 calculate_netmask(std::size_t prefix_length);

    /// GET prefix length.
    ASIO_DECL std::size_t prefix_length() const;
    
    /// Construct an CIDR address from string in CIDR notation.
    static cidr_address_v4 from_string(const char* str);
    
    /// Get the CIDR address as address in dotted decimal format.
    ASIO_DECL std::string to_string() const;

    /// Test if CIDR address contains a valid host address.
    ASIO_DECL bool is_host() const;

    /// Test if CIDR address is a real subnet of other CIDR address.
    ASIO_DECL bool is_subnet_of(const cidr_address_v4&) const;
   
    /// Get network part as CIDR address. 
    ASIO_DECL cidr_address_v4 network_cidr() const;

    /// Get host part as CIDR address. 
    ASIO_DECL cidr_address_v4 host_cidr() const;

    /// Compare two CIDR addresses for equality.
    friend bool operator==(const cidr_address_v4& a1, const cidr_address_v4& a2)
    {
      return a1.base_address_ == a2.base_address_ &&
             a1.netmask_ == a2.netmask_ &&
             a1.network_ == a2.network_ &&
             a1.broadcast_ == a2.broadcast_;
    }
    
    /// Compare two CIDR addresses for inequality.
    friend bool operator!=(const cidr_address_v4& a1, const cidr_address_v4& a2)
    {
      return ! ( a1 == a2);
    }

private:
    address_v4 base_address_;
    address_v4 netmask_;
    address_v4 network_;
    address_v4 broadcast_;
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ip/impl/cidr_address_v4.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_IP_CIDR_ADDRESS_V4_HPP
