//
// socket_types.hpp
// ~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SOCKET_TYPES_HPP
#define ASIO_DETAIL_SOCKET_TYPES_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#if defined(_WIN32)
# define FD_SETSIZE 1024
# include <winsock2.h>
#else
# include <sys/types.h>
# include <sys/select.h>
# include <sys/socket.h>
# include <netinet/in.h>
# include <netinet/tcp.h>
# include <arpa/inet.h>
# include <netdb.h>
#endif
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

#if defined(_WIN32)
typedef SOCKET socket_type;
const SOCKET invalid_socket = INVALID_SOCKET;
const int socket_error_retval = SOCKET_ERROR;
typedef sockaddr socket_addr_type;
typedef sockaddr_in inet_addr_v4_type;
typedef int socket_addr_len_type;
typedef int socket_len_type;
typedef unsigned long ioctl_arg_type;
#else
typedef int socket_type;
const int invalid_socket = -1;
const int socket_error_retval = -1;
typedef sockaddr socket_addr_type;
typedef sockaddr_in inet_addr_v4_type;
typedef socklen_t socket_addr_len_type;
typedef socklen_t socket_len_type;
typedef int ioctl_arg_type;
#endif

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_TYPES_HPP
