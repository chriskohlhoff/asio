//
// error.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ERROR_HPP
#define ASIO_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cerrno>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error_code.hpp"
#include "asio/detail/socket_types.hpp"

#if defined(GENERATING_DOCUMENTATION)
/// INTERNAL ONLY.
# define ASIO_NATIVE_ERROR(e) implementation_defined
/// INTERNAL ONLY.
# define ASIO_SOCKET_ERROR(e) implementation_defined
/// INTERNAL ONLY.
# define ASIO_NETDB_ERROR(e) implementation_defined
/// INTERNAL ONLY.
# define ASIO_GETADDRINFO_ERROR(e) implementation_defined
/// INTERNAL ONLY.
# define ASIO_WIN_OR_POSIX(e_win, e_posix) implementation_defined
#elif defined(BOOST_WINDOWS) || defined(__CYGWIN__)
# define ASIO_NATIVE_ERROR(e) \
    asio::error_code(e, \
        asio::native_ecat)
# define ASIO_SOCKET_ERROR(e) \
    asio::error_code(WSA ## e, \
        asio::native_ecat)
# define ASIO_NETDB_ERROR(e) \
    asio::error_code(WSA ## e, \
        asio::native_ecat)
# define ASIO_GETADDRINFO_ERROR(e) \
    asio::error_code(WSA ## e, \
        asio::native_ecat)
# define ASIO_MISC_ERROR(e) \
    asio::error_code(e, \
        asio::misc_ecat)
# define ASIO_WIN_OR_POSIX(e_win, e_posix) e_win
#else
# define ASIO_NATIVE_ERROR(e) \
    asio::error_code(e, \
        asio::native_ecat)
# define ASIO_SOCKET_ERROR(e) \
    asio::error_code(e, \
        asio::native_ecat)
# define ASIO_NETDB_ERROR(e) \
    asio::error_code(e, \
        asio::netdb_ecat)
# define ASIO_GETADDRINFO_ERROR(e) \
    asio::error_code(e, \
        asio::addrinfo_ecat)
# define ASIO_MISC_ERROR(e) \
    asio::error_code(e, \
        asio::misc_ecat)
# define ASIO_WIN_OR_POSIX(e_win, e_posix) e_posix
#endif

namespace asio {

namespace detail {

/// Hack to keep asio library header-file-only.
template <typename T>
class error_base
{
public:
  // boostify: error category declarations go here.

  /// Permission denied.
  static const asio::error_code access_denied;

  /// Address family not supported by protocol.
  static const asio::error_code address_family_not_supported;

  /// Address already in use.
  static const asio::error_code address_in_use;

  /// Transport endpoint is already connected.
  static const asio::error_code already_connected;

  /// Already open.
  static const asio::error_code already_open;

  /// Operation already in progress.
  static const asio::error_code already_started;

  /// A connection has been aborted.
  static const asio::error_code connection_aborted;

  /// Connection refused.
  static const asio::error_code connection_refused;

  /// Connection reset by peer.
  static const asio::error_code connection_reset;

  /// Bad file descriptor.
  static const asio::error_code bad_descriptor;

  /// End of file or stream.
  static const asio::error_code eof;

  /// Bad address.
  static const asio::error_code fault;

  /// Host not found (authoritative).
  static const asio::error_code host_not_found;

  /// Host not found (non-authoritative).
  static const asio::error_code host_not_found_try_again;

  /// No route to host.
  static const asio::error_code host_unreachable;

  /// Operation now in progress.
  static const asio::error_code in_progress;

  /// Interrupted system call.
  static const asio::error_code interrupted;

  /// Invalid argument.
  static const asio::error_code invalid_argument;

  /// Message too long.
  static const asio::error_code message_size;

  /// Network is down.
  static const asio::error_code network_down;

  /// Network dropped connection on reset.
  static const asio::error_code network_reset;

  /// Network is unreachable.
  static const asio::error_code network_unreachable;

  /// Too many open files.
  static const asio::error_code no_descriptors;

  /// No buffer space available.
  static const asio::error_code no_buffer_space;

  /// The query is valid but does not have associated address data.
  static const asio::error_code no_data;

  /// Cannot allocate memory.
  static const asio::error_code no_memory;

  /// Operation not permitted.
  static const asio::error_code no_permission;

  /// Protocol not available.
  static const asio::error_code no_protocol_option;

  /// A non-recoverable error occurred.
  static const asio::error_code no_recovery;

  /// Transport endpoint is not connected.
  static const asio::error_code not_connected;

  /// Element not found.
  static const asio::error_code not_found;

  /// Socket operation on non-socket.
  static const asio::error_code not_socket;

  /// Operation cancelled.
  static const asio::error_code operation_aborted;

  /// Operation not supported.
  static const asio::error_code operation_not_supported;

  /// The service is not supported for the given socket type.
  static const asio::error_code service_not_found;

  /// The socket type is not supported.
  static const asio::error_code socket_type_not_supported;

  /// Cannot send after transport endpoint shutdown.
  static const asio::error_code shut_down;

  /// Connection timed out.
  static const asio::error_code timed_out;

  /// Resource temporarily unavailable.
  static const asio::error_code try_again;

  /// The socket is marked non-blocking and the requested operation would block.
  static const asio::error_code would_block;

private:
  error_base();
};

// boostify: error category definitions go here.

template <typename T> const asio::error_code
error_base<T>::access_denied = ASIO_SOCKET_ERROR(EACCES);

template <typename T> const asio::error_code
error_base<T>::address_family_not_supported = ASIO_SOCKET_ERROR(
    EAFNOSUPPORT);

template <typename T> const asio::error_code
error_base<T>::address_in_use = ASIO_SOCKET_ERROR(EADDRINUSE);

template <typename T> const asio::error_code
error_base<T>::already_connected = ASIO_SOCKET_ERROR(EISCONN);

template <typename T> const asio::error_code
error_base<T>::already_open = ASIO_MISC_ERROR(1);

template <typename T> const asio::error_code
error_base<T>::already_started = ASIO_SOCKET_ERROR(EALREADY);

template <typename T> const asio::error_code
error_base<T>::connection_aborted = ASIO_SOCKET_ERROR(ECONNABORTED);

template <typename T> const asio::error_code
error_base<T>::connection_refused = ASIO_SOCKET_ERROR(ECONNREFUSED);

template <typename T> const asio::error_code
error_base<T>::connection_reset = ASIO_SOCKET_ERROR(ECONNRESET);

template <typename T> const asio::error_code
error_base<T>::bad_descriptor = ASIO_SOCKET_ERROR(EBADF);

template <typename T> const asio::error_code
error_base<T>::eof = ASIO_MISC_ERROR(2);

template <typename T> const asio::error_code
error_base<T>::fault = ASIO_SOCKET_ERROR(EFAULT);

template <typename T> const asio::error_code
error_base<T>::host_not_found = ASIO_NETDB_ERROR(HOST_NOT_FOUND);

template <typename T> const asio::error_code
error_base<T>::host_not_found_try_again = ASIO_NETDB_ERROR(TRY_AGAIN);

template <typename T> const asio::error_code
error_base<T>::host_unreachable = ASIO_SOCKET_ERROR(EHOSTUNREACH);

template <typename T> const asio::error_code
error_base<T>::in_progress = ASIO_SOCKET_ERROR(EINPROGRESS);

template <typename T> const asio::error_code
error_base<T>::interrupted = ASIO_SOCKET_ERROR(EINTR);

template <typename T> const asio::error_code
error_base<T>::invalid_argument = ASIO_SOCKET_ERROR(EINVAL);

template <typename T> const asio::error_code
error_base<T>::message_size = ASIO_SOCKET_ERROR(EMSGSIZE);

template <typename T> const asio::error_code
error_base<T>::network_down = ASIO_SOCKET_ERROR(ENETDOWN);

template <typename T> const asio::error_code
error_base<T>::network_reset = ASIO_SOCKET_ERROR(ENETRESET);

template <typename T> const asio::error_code
error_base<T>::network_unreachable = ASIO_SOCKET_ERROR(ENETUNREACH);

template <typename T> const asio::error_code
error_base<T>::no_descriptors = ASIO_SOCKET_ERROR(EMFILE);

template <typename T> const asio::error_code
error_base<T>::no_buffer_space = ASIO_SOCKET_ERROR(ENOBUFS);

template <typename T> const asio::error_code
error_base<T>::no_data = ASIO_NETDB_ERROR(NO_DATA);

template <typename T> const asio::error_code
error_base<T>::no_memory = ASIO_WIN_OR_POSIX(
    ASIO_NATIVE_ERROR(ERROR_OUTOFMEMORY),
    ASIO_NATIVE_ERROR(ENOMEM));

template <typename T> const asio::error_code
error_base<T>::no_permission = ASIO_WIN_OR_POSIX(
    ASIO_NATIVE_ERROR(ERROR_ACCESS_DENIED),
    ASIO_NATIVE_ERROR(EPERM));

template <typename T> const asio::error_code
error_base<T>::no_protocol_option = ASIO_SOCKET_ERROR(ENOPROTOOPT);

template <typename T> const asio::error_code
error_base<T>::no_recovery = ASIO_NETDB_ERROR(NO_RECOVERY);

template <typename T> const asio::error_code
error_base<T>::not_connected = ASIO_SOCKET_ERROR(ENOTCONN);

template <typename T> const asio::error_code
error_base<T>::not_found = ASIO_MISC_ERROR(3);

template <typename T> const asio::error_code
error_base<T>::not_socket = ASIO_SOCKET_ERROR(ENOTSOCK);

template <typename T> const asio::error_code
error_base<T>::operation_aborted = ASIO_WIN_OR_POSIX(
    ASIO_NATIVE_ERROR(ERROR_OPERATION_ABORTED),
    ASIO_NATIVE_ERROR(ECANCELED));

template <typename T> const asio::error_code
error_base<T>::operation_not_supported = ASIO_SOCKET_ERROR(EOPNOTSUPP);

template <typename T> const asio::error_code
error_base<T>::service_not_found = ASIO_WIN_OR_POSIX(
    ASIO_NATIVE_ERROR(WSATYPE_NOT_FOUND),
    ASIO_GETADDRINFO_ERROR(EAI_SERVICE));

template <typename T> const asio::error_code
error_base<T>::socket_type_not_supported = ASIO_WIN_OR_POSIX(
    ASIO_NATIVE_ERROR(WSAESOCKTNOSUPPORT),
    ASIO_GETADDRINFO_ERROR(EAI_SOCKTYPE));

template <typename T> const asio::error_code
error_base<T>::shut_down = ASIO_SOCKET_ERROR(ESHUTDOWN);

template <typename T> const asio::error_code
error_base<T>::timed_out = ASIO_SOCKET_ERROR(ETIMEDOUT);

template <typename T> const asio::error_code
error_base<T>::try_again = ASIO_WIN_OR_POSIX(
    ASIO_NATIVE_ERROR(ERROR_RETRY),
    ASIO_NATIVE_ERROR(EAGAIN));

template <typename T> const asio::error_code
error_base<T>::would_block = ASIO_SOCKET_ERROR(EWOULDBLOCK);

} // namespace detail

/// Contains error constants.
class error : public asio::detail::error_base<error>
{
private:
  error();
};

} // namespace asio

#undef ASIO_NATIVE_ERROR
#undef ASIO_SOCKET_ERROR
#undef ASIO_NETDB_ERROR
#undef ASIO_GETADDRINFO_ERROR
#undef ASIO_MISC_ERROR
#undef ASIO_WIN_OR_POSIX

#include "asio/impl/error_code.ipp"

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ERROR_HPP
