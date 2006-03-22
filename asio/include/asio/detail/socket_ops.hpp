//
// socket_ops.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SOCKET_OPS_HPP
#define ASIO_DETAIL_SOCKET_OPS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <boost/detail/workaround.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {
namespace socket_ops {

inline int get_error()
{
#if defined(BOOST_WINDOWS)
  return WSAGetLastError();
#else // defined(BOOST_WINDOWS)
  return errno;
#endif // defined(BOOST_WINDOWS)
}

inline void set_error(int error)
{
  errno = error;
#if defined(BOOST_WINDOWS)
  WSASetLastError(error);
#endif // defined(BOOST_WINDOWS)
}

template <typename ReturnType>
inline ReturnType error_wrapper(ReturnType return_value)
{
#if defined(BOOST_WINDOWS)
  errno = WSAGetLastError();
#endif // defined(BOOST_WINDOWS)
  return return_value;
}

inline socket_type accept(socket_type s, socket_addr_type* addr,
    socket_addr_len_type* addrlen)
{
  set_error(0);
  return error_wrapper(::accept(s, addr, addrlen));
}

inline int bind(socket_type s, const socket_addr_type* addr,
    socket_addr_len_type addrlen)
{
  set_error(0);
  return error_wrapper(::bind(s, addr, addrlen));
}

inline int close(socket_type s)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  return error_wrapper(::closesocket(s));
#else // defined(BOOST_WINDOWS)
  return error_wrapper(::close(s));
#endif // defined(BOOST_WINDOWS)
}

inline int shutdown(socket_type s, int what)
{
  set_error(0);
  return error_wrapper(::shutdown(s, what));
}

inline int connect(socket_type s, const socket_addr_type* addr,
    socket_addr_len_type addrlen)
{
  set_error(0);
  return error_wrapper(::connect(s, addr, addrlen));
}

inline int listen(socket_type s, int backlog)
{
  set_error(0);
  return error_wrapper(::listen(s, backlog));
}

#if defined(BOOST_WINDOWS)
typedef WSABUF buf;
#else // defined(BOOST_WINDOWS)
typedef iovec buf;
#endif // defined(BOOST_WINDOWS)

inline void init_buf(buf& b, void* data, size_t size)
{
#if defined(BOOST_WINDOWS)
  b.buf = static_cast<char*>(data);
  b.len = static_cast<u_long>(size);
#else // defined(BOOST_WINDOWS)
  b.iov_base = data;
  b.iov_len = size;
#endif // defined(BOOST_WINDOWS)
}

inline void init_buf(buf& b, const void* data, size_t size)
{
#if defined(BOOST_WINDOWS)
  b.buf = static_cast<char*>(const_cast<void*>(data));
  b.len = static_cast<u_long>(size);
#else // defined(BOOST_WINDOWS)
  b.iov_base = const_cast<void*>(data);
  b.iov_len = size;
#endif // defined(BOOST_WINDOWS)
}

inline int recv(socket_type s, buf* bufs, size_t count, int flags)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  // Receive some data.
  DWORD recv_buf_count = static_cast<DWORD>(count);
  DWORD bytes_transferred = 0;
  DWORD recv_flags = flags;
  int result = error_wrapper(::WSARecv(s, bufs,
        recv_buf_count, &bytes_transferred, &recv_flags, 0, 0));
  if (result != 0)
    return -1;
  return bytes_transferred;
#else // defined(BOOST_WINDOWS)
  msghdr msg;
  msg.msg_name = 0;
  msg.msg_namelen = 0;
  msg.msg_iov = bufs;
  msg.msg_iovlen = count;
  msg.msg_control = 0;
  msg.msg_controllen = 0;
  msg.msg_flags = 0;
  return error_wrapper(::recvmsg(s, &msg, flags));
#endif // defined(BOOST_WINDOWS)
}

inline int recvfrom(socket_type s, buf* bufs, size_t count, int flags,
    socket_addr_type* addr, socket_addr_len_type* addrlen)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  // Receive some data.
  DWORD recv_buf_count = static_cast<DWORD>(count);
  DWORD bytes_transferred = 0;
  DWORD recv_flags = flags;
  int result = error_wrapper(::WSARecvFrom(s, bufs, recv_buf_count,
        &bytes_transferred, &recv_flags, addr, addrlen, 0, 0));
  if (result != 0)
    return -1;
  return bytes_transferred;
#else // defined(BOOST_WINDOWS)
  msghdr msg;
  msg.msg_name = addr;
  msg.msg_namelen = *addrlen;
  msg.msg_iov = bufs;
  msg.msg_iovlen = count;
  msg.msg_control = 0;
  msg.msg_controllen = 0;
  msg.msg_flags = 0;
  int result = error_wrapper(::recvmsg(s, &msg, flags));
  *addrlen = msg.msg_namelen;
  return result;
#endif // defined(BOOST_WINDOWS)
}

inline int send(socket_type s, const buf* bufs, size_t count, int flags)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  // Send the data.
  DWORD send_buf_count = static_cast<DWORD>(count);
  DWORD bytes_transferred = 0;
  DWORD send_flags = flags;
  int result = error_wrapper(::WSASend(s, const_cast<buf*>(bufs),
        send_buf_count, &bytes_transferred, send_flags, 0, 0));
  if (result != 0)
    return -1;
  return bytes_transferred;
#else // defined(BOOST_WINDOWS)
  msghdr msg;
  msg.msg_name = 0;
  msg.msg_namelen = 0;
  msg.msg_iov = const_cast<buf*>(bufs);
  msg.msg_iovlen = count;
  msg.msg_control = 0;
  msg.msg_controllen = 0;
  msg.msg_flags = 0;
  return error_wrapper(::sendmsg(s, &msg, flags));
#endif // defined(BOOST_WINDOWS)
}

inline int sendto(socket_type s, const buf* bufs, size_t count, int flags,
    const socket_addr_type* addr, socket_addr_len_type addrlen)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  // Send the data.
  DWORD send_buf_count = static_cast<DWORD>(count);
  DWORD bytes_transferred = 0;
  int result = ::WSASendTo(s, const_cast<buf*>(bufs), send_buf_count,
      &bytes_transferred, flags, addr, addrlen, 0, 0);
  if (result != 0)
    return -1;
  return bytes_transferred;
#else // defined(BOOST_WINDOWS)
  msghdr msg;
  msg.msg_name = const_cast<socket_addr_type*>(addr);
  msg.msg_namelen = addrlen;
  msg.msg_iov = const_cast<buf*>(bufs);
  msg.msg_iovlen = count;
  msg.msg_control = 0;
  msg.msg_controllen = 0;
  msg.msg_flags = 0;
  return error_wrapper(::sendmsg(s, &msg, flags));
#endif // defined(BOOST_WINDOWS)
}

inline socket_type socket(int af, int type, int protocol)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  return error_wrapper(::WSASocket(af, type, protocol, 0, 0,
        WSA_FLAG_OVERLAPPED));
#else // defined(BOOST_WINDOWS)
  return error_wrapper(::socket(af, type, protocol));
#endif // defined(BOOST_WINDOWS)
}

inline int setsockopt(socket_type s, int level, int optname,
    const void* optval, size_t optlen)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  return error_wrapper(::setsockopt(s, level, optname,
        reinterpret_cast<const char*>(optval), static_cast<int>(optlen)));
#else // defined(BOOST_WINDOWS)
  return error_wrapper(::setsockopt(s, level, optname, optval,
        static_cast<socklen_t>(optlen)));
#endif // defined(BOOST_WINDOWS)
}

inline int getsockopt(socket_type s, int level, int optname, void* optval,
    size_t* optlen)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  int tmp_optlen = static_cast<int>(*optlen);
  int result = error_wrapper(::getsockopt(s, level, optname,
        reinterpret_cast<char*>(optval), &tmp_optlen));
  *optlen = static_cast<size_t>(tmp_optlen);
  return result;
#else // defined(BOOST_WINDOWS)
  socklen_t tmp_optlen = static_cast<socklen_t>(*optlen);
  int result = error_wrapper(::getsockopt(s, level, optname,
        optval, &tmp_optlen));
  *optlen = static_cast<size_t>(tmp_optlen);
  return result;
#endif // defined(BOOST_WINDOWS)
}

inline int getpeername(socket_type s, socket_addr_type* addr,
    socket_addr_len_type* addrlen)
{
  set_error(0);
  return error_wrapper(::getpeername(s, addr, addrlen));
}

inline int getsockname(socket_type s, socket_addr_type* addr,
    socket_addr_len_type* addrlen)
{
  set_error(0);
  return error_wrapper(::getsockname(s, addr, addrlen));
}

inline int ioctl(socket_type s, long cmd, ioctl_arg_type* arg)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  return error_wrapper(::ioctlsocket(s, cmd, arg));
#else // defined(BOOST_WINDOWS)
  return error_wrapper(::ioctl(s, cmd, arg));
#endif // defined(BOOST_WINDOWS)
}

inline int select(int nfds, fd_set* readfds, fd_set* writefds,
    fd_set* exceptfds, timeval* timeout)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  if (!readfds && !writefds && !exceptfds && timeout)
  {
    DWORD milliseconds = timeout->tv_sec * 1000 + timeout->tv_usec / 1000;
    if (milliseconds == 0)
      milliseconds = 1; // Force context switch.
    ::Sleep(milliseconds);
    return 0;
  }
#endif // defined(BOOST_WINDOWS)
  return error_wrapper(::select(nfds, readfds, writefds, exceptfds, timeout));
}

inline int poll_read(socket_type s)
{
#if defined(BOOST_WINDOWS)
  FD_SET fds;
  FD_ZERO(&fds);
  FD_SET(s, &fds);
  set_error(0);
  return error_wrapper(::select(s, &fds, 0, 0, 0));
#else // defined(BOOST_WINDOWS)
  pollfd fds;
  fds.fd = s;
  fds.events = POLLIN;
  fds.revents = 0;
  set_error(0);
  return error_wrapper(::poll(&fds, 1, -1));
#endif // defined(BOOST_WINDOWS)
}

inline int poll_write(socket_type s)
{
#if defined(BOOST_WINDOWS)
  FD_SET fds;
  FD_ZERO(&fds);
  FD_SET(s, &fds);
  set_error(0);
  return error_wrapper(::select(s, 0, &fds, 0, 0));
#else // defined(BOOST_WINDOWS)
  pollfd fds;
  fds.fd = s;
  fds.events = POLLOUT;
  fds.revents = 0;
  set_error(0);
  return error_wrapper(::poll(&fds, 1, -1));
#endif // defined(BOOST_WINDOWS)
}

inline const char* inet_ntop(int af, const void* src, char* dest, size_t length,
    unsigned long scope_id = 0)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  using namespace std; // For memcpy.

  if (af != AF_INET && af != AF_INET6)
  {
    set_error(asio::error::address_family_not_supported);
    return 0;
  }

  sockaddr_storage address;
  DWORD address_length;
  if (af == AF_INET)
  {
    address_length = sizeof(sockaddr_in);
    sockaddr_in* ipv4_address = reinterpret_cast<sockaddr_in*>(&address);
    ipv4_address->sin_family = AF_INET;
    ipv4_address->sin_port = 0;
    memcpy(&ipv4_address->sin_addr, src, sizeof(in_addr));
  }
  else // AF_INET6
  {
    address_length = sizeof(sockaddr_in6);
    sockaddr_in6* ipv6_address = reinterpret_cast<sockaddr_in6*>(&address);
    ipv6_address->sin6_family = AF_INET6;
    ipv6_address->sin6_port = 0;
    ipv6_address->sin6_flowinfo = 0;
    ipv6_address->sin6_scope_id = scope_id;
    memcpy(&ipv6_address->sin6_addr, src, sizeof(in6_addr));
  }

  DWORD string_length = length;
  int result = error_wrapper(::WSAAddressToStringA(
        reinterpret_cast<sockaddr*>(&address),
        address_length, 0, dest, &string_length));

  // Windows may not set an error code on failure.
  if (result == socket_error_retval && get_error() == 0)
    set_error(asio::error::invalid_argument);

  return result == socket_error_retval ? 0 : dest;
#else // defined(BOOST_WINDOWS)
  const char* result = error_wrapper(::inet_ntop(af, src, dest, length));
  if (result == 0 && get_error() == 0)
    set_error(asio::error::invalid_argument);
  if (result != 0 && af == AF_INET6 && scope_id != 0)
  {
    using namespace std; // For strcat and sprintf.
    char if_name[IF_NAMESIZE + 1] = "%";
    const in6_addr* ipv6_address = static_cast<const in6_addr*>(src);
    bool is_link_local = IN6_IS_ADDR_LINKLOCAL(ipv6_address);
    if (!is_link_local || if_indextoname(scope_id, if_name + 1) == 0)
      sprintf(if_name + 1, "%lu", scope_id);
    strcat(dest, if_name);
  }
  return result;
#endif // defined(BOOST_WINDOWS)
}

inline int inet_pton(int af, const char* src, void* dest,
    unsigned long* scope_id = 0)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  using namespace std; // For memcpy and strcmp.

  if (af != AF_INET && af != AF_INET6)
  {
    set_error(asio::error::address_family_not_supported);
    return -1;
  }

  sockaddr_storage address;
  int address_length = sizeof(sockaddr_storage);
  int result = error_wrapper(::WSAStringToAddressA(
        const_cast<char*>(src), af, 0,
        reinterpret_cast<sockaddr*>(&address),
        &address_length));

  if (af == AF_INET)
  {
    if (result != socket_error_retval)
    {
      sockaddr_in* ipv4_address = reinterpret_cast<sockaddr_in*>(&address);
      memcpy(dest, &ipv4_address->sin_addr, sizeof(in_addr));
    }
    else if (strcmp(src, "255.255.255.255") == 0)
    {
      static_cast<in_addr*>(dest)->s_addr = INADDR_NONE;
    }
  }
  else // AF_INET6
  {
    if (result != socket_error_retval)
    {
      sockaddr_in6* ipv6_address = reinterpret_cast<sockaddr_in6*>(&address);
      memcpy(dest, &ipv6_address->sin6_addr, sizeof(in6_addr));
      if (scope_id)
        *scope_id = ipv6_address->sin6_scope_id;
    }
  }

  // Windows may not set an error code on failure.
  if (result == socket_error_retval && get_error() == 0)
    set_error(asio::error::invalid_argument);

  return result == socket_error_retval ? -1 : 1;
#else // defined(BOOST_WINDOWS)
  int result = error_wrapper(::inet_pton(af, src, dest));
  if (result <= 0 && get_error() == 0)
    set_error(asio::error::invalid_argument);
  if (result > 0 && af == AF_INET6 && scope_id)
  {
    using namespace std; // For strchr and atoi.
    *scope_id = 0;
    if (const char* if_name = strchr(src, '%'))
    {
      in6_addr* ipv6_address = static_cast<in6_addr*>(dest);
      bool is_link_local = IN6_IS_ADDR_LINKLOCAL(ipv6_address);
      if (is_link_local)
        *scope_id = if_nametoindex(if_name + 1);
      if (*scope_id == 0)
        *scope_id = atoi(if_name + 1);
    }
  }
  return result;
#endif // defined(BOOST_WINDOWS)
}

inline int gethostname(char* name, int namelen)
{
  set_error(0);
  return error_wrapper(::gethostname(name, namelen));
}

inline int translate_netdb_error(int error)
{
  switch (error)
  {
  case 0:
    return asio::error::success;
  case HOST_NOT_FOUND:
    return asio::error::host_not_found;
  case TRY_AGAIN:
    return asio::error::host_not_found_try_again;
  case NO_RECOVERY:
    return asio::error::no_recovery;
  case NO_DATA:
    return asio::error::no_host_data;
  default:
    BOOST_ASSERT(false);
    return get_error();
  }
}

inline hostent* gethostbyaddr(const char* addr, int length, int type,
    hostent* result, char* buffer, int buflength, int* error)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  hostent* retval = error_wrapper(::gethostbyaddr(addr, length, type));
  *error = get_error();
  if (!retval)
    return 0;
  *result = *retval;
  return retval;
#elif defined(__sun)
  hostent* retval = error_wrapper(::gethostbyaddr_r(addr, length, type, result,
        buffer, buflength, error));
  *error = translate_netdb_error(*error);
  return retval;
#elif defined(__MACH__) && defined(__APPLE__)
  hostent* retval = error_wrapper(::getipnodebyaddr(addr, length, type, error));
  *error = translate_netdb_error(*error);
  if (!retval)
    return 0;
  *result = *retval;
  return retval;
#else
  hostent* retval = 0;
  error_wrapper(::gethostbyaddr_r(addr, length, type, result, buffer,
        buflength, &retval, error));
  *error = translate_netdb_error(*error);
  return retval;
#endif
}

inline hostent* gethostbyname(const char* name, struct hostent* result,
    char* buffer, int buflength, int* error)
{
  set_error(0);
#if defined(BOOST_WINDOWS)
  hostent* retval = error_wrapper(::gethostbyname(name));
  *error = get_error();
  if (!retval)
    return 0;
  *result = *retval;
  return result;
#elif defined(__sun)
  hostent* retval = error_wrapper(::gethostbyname_r(name, result, buffer,
        buflength, error));
  *error = translate_netdb_error(*error);
  return retval;
#elif defined(__MACH__) && defined(__APPLE__)
  hostent* retval = error_wrapper(::getipnodebyname(name, AF_INET, 0, error));
  *error = translate_netdb_error(*error);
  if (!retval)
    return 0;
  *result = *retval;
  return retval;
#else
  hostent* retval = 0;
  error_wrapper(::gethostbyname_r(name, result, buffer, buflength, &retval,
        error));
  *error = translate_netdb_error(*error);
  return retval;
#endif
}

inline void freehostent(hostent* h)
{
#if defined(__MACH__) && defined(__APPLE__)
  ::freehostent(h);
#endif
}

inline u_long_type network_to_host_long(u_long_type value)
{
  return ntohl(value);
}

inline u_long_type host_to_network_long(u_long_type value)
{
  return htonl(value);
}

inline u_short_type network_to_host_short(u_short_type value)
{
  return ntohs(value);
}

inline u_short_type host_to_network_short(u_short_type value)
{
  return htons(value);
}

} // namespace socket_ops
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_OPS_HPP
