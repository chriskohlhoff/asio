//
// socket_types.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SOCKET_TYPES_HPP
#define ASIO_DETAIL_SOCKET_TYPES_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#if defined(_WIN32)
# if defined(__BORLANDC__) && !defined(_WSPIAPI_H_)
#  include <stdlib.h> // Needed for __errno
#  define _WSPIAPI_H_
#  define ASIO_WSPIAPI_H_DEFINED
# endif // defined(__BORLANDC__) && !defined(_WSPIAPI_H_)
# define FD_SETSIZE 1024
# include <winsock2.h>
# include <ws2tcpip.h>
# if defined(ASIO_WSPIAPI_H_DEFINED)
#  undef _WSPIAPI_H_
#  undef ASIO_WSPIAPI_H_DEFINED
# endif // defined(ASIO_WSPIAPI_H_DEFINED)
#else
# include <sys/ioctl.h>
# include <sys/types.h>
# include <sys/select.h>
# include <sys/socket.h>
# include <netinet/in.h>
# include <netinet/tcp.h>
# include <arpa/inet.h>
# include <netdb.h>
# if defined(__sun)
#  include <sys/filio.h>
# endif
#endif
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

#if defined(_WIN32)
typedef SOCKET socket_type;
const SOCKET invalid_socket = INVALID_SOCKET;
const int socket_error_retval = SOCKET_ERROR;
const int max_addr_str_len = 256;
typedef sockaddr socket_addr_type;
typedef sockaddr_in inet_addr_v4_type;
typedef int socket_addr_len_type;
typedef unsigned long ioctl_arg_type;
typedef u_long u_long_type;
typedef u_short u_short_type;
const int shutdown_receive = SD_RECEIVE;
const int shutdown_send = SD_SEND;
const int shutdown_both = SD_BOTH;
const int message_peek = MSG_PEEK;
const int message_out_of_band = MSG_OOB;
const int message_do_not_route = MSG_DONTROUTE;
#else
typedef int socket_type;
const int invalid_socket = -1;
const int socket_error_retval = -1;
const int max_addr_str_len = INET_ADDRSTRLEN;
typedef sockaddr socket_addr_type;
typedef sockaddr_in inet_addr_v4_type;
typedef socklen_t socket_addr_len_type;
typedef int ioctl_arg_type;
typedef uint32_t u_long_type;
typedef uint16_t u_short_type;
const int shutdown_receive = SHUT_RD;
const int shutdown_send = SHUT_WR;
const int shutdown_both = SHUT_RDWR;
const int message_peek = MSG_PEEK;
const int message_out_of_band = MSG_OOB;
const int message_do_not_route = MSG_DONTROUTE;
#endif

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_TYPES_HPP
