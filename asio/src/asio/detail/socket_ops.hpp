//
// socket_ops.hpp
// ~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SOCKET_OPS_HPP
#define ASIO_DETAIL_SOCKET_OPS_HPP

#include <cerrno>
#include "asio/detail/socket_types.hpp"
#if !defined(_WIN32)
#include <sys/ioctl.h>
#endif

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {
namespace socket_ops {

inline
int
get_error()
{
#if defined(_WIN32)
  return WSAGetLastError();
#else // defined(_WIN32)
  return errno;
#endif // defined(_WIN32)
}

inline
void
set_error(
    int error)
{
  errno = error;
#if defined(_WIN32)
  WSASetLastError(error);
#endif // defined(_WIN32)
}

template <typename ReturnType>
inline
ReturnType
error_wrapper(
    ReturnType return_value)
{
#if defined(_WIN32)
  errno = WSAGetLastError();
#endif // defined(_WIN32)
  return return_value;
}

inline
socket_type
accept(
    socket_type s,
    socket_addr_type* addr,
    socket_addr_len_type* addrlen)
{
  set_error(0);
  return error_wrapper(::accept(s, addr, addrlen));
}

inline
int
bind(
    socket_type s,
    const socket_addr_type* addr,
    socket_addr_len_type addrlen)
{
  set_error(0);
  return error_wrapper(::bind(s, addr, addrlen));
}

inline
void
close(
    socket_type s)
{
  set_error(0);
#if defined(_WIN32)
  error_wrapper(::closesocket(s));
#else // defined(_WIN32)
  error_wrapper(::close(s));
#endif // defined(_WIN32)
}

inline
int
connect(
    socket_type s,
    const socket_addr_type* addr,
    socket_addr_len_type addrlen)
{
  set_error(0);
  return error_wrapper(::connect(s, addr, addrlen));
}

inline
int
listen(
    socket_type s,
    int backlog)
{
  set_error(0);
  return error_wrapper(::listen(s, backlog));
}

inline
int
recv(
    socket_type s,
    void* buf,
    size_t len,
    int flags)
{
  set_error(0);
#if defined(_WIN32)
  return error_wrapper(::recv(s, static_cast<char*>(buf), len, flags));
#else // defined(_WIN32)
  return error_wrapper(::recv(s, buf, len, flags));
#endif // defined(_WIN32)
}

inline
int
recvfrom(
    socket_type s,
    void* buf,
    size_t len,
    int flags,
    socket_addr_type* addr,
    socket_addr_len_type* addrlen)
{
  set_error(0);
#if defined(_WIN32)
  return error_wrapper(::recvfrom(s, static_cast<char*>(buf), len, flags, addr,
        addrlen));
#else // defined(_WIN32)
  return error_wrapper(::recvfrom(s, buf, len, flags, addr, addrlen));
#endif // defined(_WIN32)
}

inline
int
send(
    socket_type s,
    const void* buf,
    size_t len,
    int flags)
{
  set_error(0);
#if defined(_WIN32)
  return error_wrapper(::send(s, static_cast<const char*>(buf), len, flags));
#else // defined(_WIN32)
  return error_wrapper(::send(s, buf, len, flags));
#endif // defined(_WIN32)
}

inline
int
sendto(
    socket_type s,
    const void* buf,
    size_t len,
    int flags,
    const socket_addr_type* addr,
    socket_addr_len_type addrlen)
{
  set_error(0);
#if defined(_WIN32)
  return error_wrapper(::sendto(s, static_cast<const char*>(buf), len, flags,
        addr, addrlen));
#else // defined(_WIN32)
  return error_wrapper(::sendto(s, buf, len, flags, addr, addrlen));
#endif // defined(_WIN32)
}

inline
socket_type
socket(
    int af,
    int type,
    int protocol)
{
  set_error(0);
#if defined(_WIN32)
  return error_wrapper(::WSASocket(af, type, protocol, 0, 0,
        WSA_FLAG_OVERLAPPED));
#else // defined(_WIN32)
  return error_wrapper(::socket(af, type, protocol));
#endif // defined(_WIN32)
}

inline
int
setsockopt(
    socket_type s,
    int level,
    int optname,
    const void* optval,
    socket_len_type optlen)
{
  set_error(0);
#if defined(_WIN32)
  return error_wrapper(::setsockopt(s, level, optname,
        reinterpret_cast<const char*>(optval), optlen));
#else // defined(_WIN32)
  return error_wrapper(::setsockopt(s, level, optname, optval, optlen));
#endif // defined(_WIN32)
}

inline
int
getsockopt(
    socket_type s,
    int level,
    int optname,
    void* optval,
    socket_len_type* optlen)
{
  set_error(0);
#if defined(_WIN32)
  return error_wrapper(::getsockopt(s, level, optname,
        reinterpret_cast<char*>(optval), optlen));
#else // defined(_WIN32)
  return error_wrapper(::getsockopt(s, level, optname, optval, optlen));
#endif // defined(_WIN32)
}

inline
int
ioctl(
    socket_type s,
    long cmd,
    ioctl_arg_type* arg)
{
  set_error(0);
#if defined(_WIN32)
  return error_wrapper(::ioctlsocket(s, cmd, arg));
#else // defined(_WIN32)
  return error_wrapper(::ioctl(s, cmd, arg));
#endif // defined(_WIN32)
}

} // namespace socket_ops
} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_OPS_HPP
