//
// inet_address_v4.cpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#include "asio/inet_address_v4.hpp"

#include "asio/detail/push_options.hpp"
#if !defined(_WIN32)
#include <netdb.h>
#endif
#include <string.h>
#include "asio/detail/pop_options.hpp"

namespace asio {

inet_address_v4::
inet_address_v4()
  : good_(true)
{
  addr_.sin_family = AF_INET;
  addr_.sin_port = 0;
  addr_.sin_addr.s_addr = INADDR_ANY;
}

inet_address_v4::
inet_address_v4(
    port_type port_num)
  : good_(true)
{
  addr_.sin_family = AF_INET;
  addr_.sin_port = htons(port_num);
  addr_.sin_addr.s_addr = INADDR_ANY;
}

inet_address_v4::
inet_address_v4(
    port_type port_num,
    addr_type host_addr)
  : good_(true)
{
  addr_.sin_family = AF_INET;
  addr_.sin_port = htons(port_num);
  addr_.sin_addr.s_addr = host_addr;
}

inet_address_v4::
inet_address_v4(
    port_type port_num,
    const std::string& host)
  : good_(true)
{
  addr_.sin_family = AF_INET;
  addr_.sin_port = htons(port_num);
  host_name(host);
}

inet_address_v4::
inet_address_v4(
    const inet_address_v4& other)
  : addr_(other.addr_),
    good_(other.good_) 
{
}

inet_address_v4&
inet_address_v4::
operator=(
    const inet_address_v4& other)
{
  addr_ = other.addr_;
  good_ = other.good_;
  return *this;
}

inet_address_v4::
~inet_address_v4()
{
}

bool
inet_address_v4::
good() const
{
  return good_ && native_address()->sa_family == AF_INET;
}

bool
inet_address_v4::
bad() const
{
  return !good();
}

int
inet_address_v4::
family() const
{
  return AF_INET;
}

socket_address::native_address_type*
inet_address_v4::
native_address()
{
  return reinterpret_cast<socket_address::native_address_type*>(&addr_);
}

const socket_address::native_address_type*
inet_address_v4::
native_address() const
{
  return reinterpret_cast<const socket_address::native_address_type*>(&addr_);
}

socket_address::native_size_type
inet_address_v4::
native_size() const
{
  return sizeof(addr_);
}

void
inet_address_v4::
native_size(
    native_size_type size)
{
  good_ = (size == sizeof(addr_));
}

inet_address_v4::port_type
inet_address_v4::
port() const
{
  return ntohs(addr_.sin_port);
}

void
inet_address_v4::
port(
    port_type port_num)
{
  addr_.sin_port = htons(port_num);
}

inet_address_v4::addr_type
inet_address_v4::
host_addr() const
{
  return addr_.sin_addr.s_addr;
}

void
inet_address_v4::
host_addr(
    addr_type host)
{
  addr_.sin_addr.s_addr = host;
  good_ = true;
}

std::string
inet_address_v4::
host_addr_str() const
{
#if defined(_WIN32)
  return inet_ntoa(addr_.sin_addr);
#else
  char addr_str[INET_ADDRSTRLEN];
  return inet_ntop(AF_INET, &addr_.sin_addr, addr_str, INET_ADDRSTRLEN);
#endif
}

void
inet_address_v4::
host_addr_str(
    const std::string& host)
{
#if defined(_WIN32)
  addr_.sin_addr.s_addr = inet_addr(host.c_str());
  good_ = (addr_.sin_addr.s_addr != INADDR_NONE || host == "255.255.255.255");
#else
  good_ = inet_pton(AF_INET, host.c_str(), &addr_.sin_addr);
#endif
}

std::string
inet_address_v4::
host_name() const
{
#if defined(_WIN32)
  hostent* ent_result =
    gethostbyaddr(reinterpret_cast<const char*>(&addr_.sin_addr),
        sizeof(addr_.sin_addr), AF_INET);
  return ent_result ? ent_result->h_name : "";
#else
  hostent ent;
  hostent* ent_result;
  char buf[1024] = "";
  int error;
  gethostbyaddr_r(&addr_.sin_addr, sizeof(addr_.sin_addr), AF_INET, &ent, buf,
      sizeof(buf), &ent_result, &error);
  return ent_result->h_name;
#endif
}

void
inet_address_v4::
host_name(
    const std::string& host)
{
#if defined(_WIN32)
  hostent* ent_result = gethostbyname(host.c_str());
  good_ = (ent_result != 0);
  if (good_)
    memcpy(&addr_.sin_addr, ent_result->h_addr, sizeof(addr_.sin_addr));
#else
  hostent ent;
  hostent* ent_result;
  char buf[1024] = "";
  int error;
  good_ = (gethostbyname_r(host.c_str(), &ent, buf, sizeof(buf), &ent_result,
        &error) == 0);
  if (good_)
    memcpy(&addr_.sin_addr, ent_result->h_addr, sizeof(addr_.sin_addr));
#endif
}

} // namespace asio
