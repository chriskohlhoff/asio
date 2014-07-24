#ifndef ASIO_IP_IMPL_CIDR_ADDRESS_V4_IPP
#define ASIO_IP_IMPL_CIDR_ADDRESS_V4_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/ip/cidr_address_v4.hpp"

#include <algorithm>
#include <sstream>
#include <iostream>

#include "asio/detail/throw_error.hpp"
#include "asio/detail/throw_exception.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

address_iterator_v4::address_iterator_v4(const address_v4& addr)
  : address_(addr)
{
}

const address_v4& address_iterator_v4::operator*() const
{
  return address_;
}

const address_v4* address_iterator_v4::operator->() const
{
  return &address_;
}

address_iterator_v4& address_iterator_v4::operator++()
{
  address_ = address_v4((address_.to_ulong() + 1) & 0xFFFFFFFF);
  return *this;
}

address_iterator_v4 address_iterator_v4::operator++(int)
{
  address_iterator_v4 tmp(*this);
  ++*this;
  return tmp;
}

address_iterator_v4& address_iterator_v4::operator--()
{
  address_ = address_v4((address_.to_ulong() - 1) & 0xFFFFFFFF);
  return *this;
}

address_iterator_v4 address_iterator_v4::operator--(int)
{
  address_iterator_v4 tmp(*this);
  --*this;
  return tmp;
}

bool operator==(const address_iterator_v4& a, const address_iterator_v4& b)
{
  return a.address_ == b.address_;
}

bool operator!=(const address_iterator_v4& a, const address_iterator_v4& b)
{
  return a.address_ != b.address_;
}

cidr_address_v4::cidr_address_v4()
  : base_address_(0),
    netmask_(0),
    network_(0),
    broadcast_(0xFFFFFFFF)
{
}

cidr_address_v4::cidr_address_v4(const address_v4& addr, std::size_t prefix_length)
  : base_address_( addr),
    netmask_(calculate_netmask(prefix_length)),
    network_(addr.to_ulong() & netmask_.to_ulong()),
    broadcast_(address_v4::broadcast(network_, netmask_))
{}

cidr_address_v4::cidr_address_v4(const address_v4& addr, const address_v4& mask)
  : base_address_(addr),
    netmask_(mask),
    network_(addr.to_ulong() & netmask_.to_ulong()),
    broadcast_(address_v4::broadcast(network_, netmask_))
{
}

cidr_address_v4::cidr_address_v4(const cidr_address_v4& addr)
  : base_address_(addr.base_address_),
    netmask_(addr.netmask_),
    network_(addr.network_),
    broadcast_(addr.broadcast_)
{
}

address_v4 cidr_address_v4::network() const
{
  return network_;
}

address_v4 cidr_address_v4::host() const
{
    return base_address_;
}

address_v4 cidr_address_v4::netmask() const
{
  return netmask_;
}

address_v4 cidr_address_v4::broadcast() const
{
  return broadcast_;
}

cidr_address_v4::iterator cidr_address_v4::begin() const
{
  return iterator(address_v4(network_.to_ulong() + 1));
}

cidr_address_v4::iterator cidr_address_v4::end() const
{
  return iterator(broadcast_);
}

cidr_address_v4::iterator cidr_address_v4::find(const address_v4& addr) const
{
  return addr > network_ && addr < broadcast_ ? iterator(addr) : end();
}

std::size_t cidr_address_v4::calculate_prefix_length(const address_v4& netmask)
{
    address_v4::bytes_type mask = netmask.to_bytes();
    bool finished = false;
    std::size_t nbits = 0;

    for ( std::size_t i = 0; i < mask.size(); ++i)
    {
        if ( finished)
        {
            if ( 0 != mask[i]) {
                std::out_of_range ex("prefix from netmask");
                asio::detail::throw_exception(ex);
            }
            continue;
        }
        else
        {
            switch ( mask[i])
            {
                case 255:
                    nbits += 8;
                    break;
                case 254: // nbits += 7
                    nbits += 1;
                case 252: // nbits += 6
                    nbits += 1;
                case 248: // nbits += 5
                    nbits += 1;
                case 240: // nbits += 4
                    nbits += 1;
                case 224: // nbits += 3
                    nbits += 1;
                case 192: // nbits += 2
                    nbits += 1;
                case 128: // nbits += 1
                    nbits += 1;
                case 0:   // nbits += 0
                    finished = true;
                    break;
                default:
                    std::out_of_range ex("prefix from netmask");
                    asio::detail::throw_exception(ex);
            }
        }
    }

    return nbits; 
}

address_v4 cidr_address_v4::calculate_netmask(std::size_t prefix_length)
{
    if ( 32 < prefix_length)
    {
        std::out_of_range ex("netmask from prefix");
        asio::detail::throw_exception(ex);
    }
    uint32_t nmbits = 0xffffffff;
    if ( 0 == prefix_length)
    {
        nmbits = 0;
    }
    else
    {
        nmbits = nmbits << (32 - prefix_length);
    }
    return address_v4(nmbits);
}

std::size_t cidr_address_v4::prefix_length() const
{
    return calculate_prefix_length(netmask_);
}

cidr_address_v4 cidr_address_v4::from_string(const char* str)
{
    std::string addr_str( str);
    std::string::size_type pos = addr_str.find_first_of("/");

    if (std::string::npos == pos)
    {
        std::invalid_argument ex("no prefix found");
        asio::detail::throw_exception(ex);
    }

    if (pos == ( addr_str.size() - 1))
    {
        std::invalid_argument ex("prefix is empty");
        asio::detail::throw_exception(ex);
    }

    std::string::size_type end = addr_str.find_first_not_of("0123456789",pos+1);
    if (std::string::npos != end)
    {
        std::invalid_argument ex("invalid prefix");
        asio::detail::throw_exception(ex);
    }

    return cidr_address_v4(
        address_v4::from_string( addr_str.substr( 0, pos) ),
        atoi( addr_str.substr( pos + 1).c_str() ) );
}

std::string cidr_address_v4::to_string() const
{
    std::stringstream oss;
    oss << base_address_.to_string();
    oss << "/";
    oss << prefix_length();
    return oss.str();
}

bool cidr_address_v4::is_host() const
{
    return 32 == prefix_length();
}

bool cidr_address_v4::is_subnet_of( cidr_address_v4 const& net) const
{
    if ( this == & net) {
        return false;
    }
    if ( net.prefix_length() >= prefix_length() ) {
        return false; // only real subsets are allowed.
    }
    const cidr_address_v4 me(host(), net.netmask());
    // TODO: ugly, but cidr_address_v4 has no comparation operators yet
    return net.network_cidr().to_string() == me.network_cidr().to_string();
}

cidr_address_v4 cidr_address_v4::network_cidr() const
{
    return cidr_address_v4(network_,netmask_);
}

cidr_address_v4 cidr_address_v4::host_cidr() const
{
    return cidr_address_v4(base_address_,32);
}

} // namespace ip
} // namespace asio

#endif // ASIO_IP_IMPL_CIDR_ADDRESS_V4_IPP
