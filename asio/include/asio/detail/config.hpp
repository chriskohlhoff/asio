//
// detail/config.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CONFIG_HPP
#define ASIO_DETAIL_CONFIG_HPP

#include <boost/config.hpp>

// Default to a header-only implementation. The user must specifically request
// separate compilation by defining ASIO_SEPARATE_COMPILATION.
#if !defined(ASIO_HEADER_ONLY)
# if !defined(ASIO_SEPARATE_COMPILATION)
#  define ASIO_HEADER_ONLY
# endif // !defined(ASIO_SEPARATE_COMPILATION)
#endif // !defined(ASIO_HEADER_ONLY)

#if defined(ASIO_HEADER_ONLY)
# define ASIO_DECL inline
#else // defined(ASIO_HEADER_ONLY)
# if defined(BOOST_HAS_DECLSPEC)
// We need to import/export our code only if the user has specifically asked
// for it by defining ASIO_DYN_LINK.
#  if defined(ASIO_DYN_LINK)
// Export if this is our own source, otherwise import.
#   if defined(ASIO_SOURCE)
#    define ASIO_DECL __declspec(dllexport)
#   else // defined(ASIO_SOURCE)
#    define ASIO_DECL __declspec(dllimport)
#   endif // defined(ASIO_SOURCE)
#  endif // defined(ASIO_DYN_LINK)
# endif // defined(BOOST_HAS_DECLSPEC)
#endif // defined(ASIO_HEADER_ONLY)

// If ASIO_DECL isn't defined yet define it now.
#if !defined(ASIO_DECL)
# define ASIO_DECL
#endif // !defined(ASIO_DECL)

// Windows: IO Completion Ports.
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
# if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)
#  if !defined(UNDER_CE)
#   if !defined(ASIO_DISABLE_IOCP)
#    define ASIO_HAS_IOCP 1
#   endif // !defined(ASIO_DISABLE_IOCP)
#  endif // !defined(UNDER_CE)
# endif // defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)
#endif // defined(BOOST_WINDOWS) || defined(__CYGWIN__)

// Linux: epoll, eventfd and timerfd.
#if defined(__linux__)
# include <linux/version.h>
# if !defined(ASIO_DISABLE_EPOLL)
#  if LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,45)
#   define ASIO_HAS_EPOLL 1
#  endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2,5,45)
# endif // !defined(ASIO_DISABLE_EVENTFD)
# if !defined(ASIO_DISABLE_EVENTFD)
#  if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,22)
#   define ASIO_HAS_EVENTFD 1
#  endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,22)
# endif // !defined(ASIO_DISABLE_EVENTFD)
# if defined(ASIO_HAS_EPOLL)
#  if (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 8)
#   define ASIO_HAS_TIMERFD 1
#  endif // (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 8)
# endif // defined(ASIO_HAS_EPOLL)
#endif // defined(__linux__)

// Mac OS X, FreeBSD, NetBSD, OpenBSD: kqueue.
#if (defined(__MACH__) && defined(__APPLE__)) \
  || defined(__FreeBSD__) \
  || defined(__NetBSD__) \
  || defined(__OpenBSD__)
# if !defined(ASIO_DISABLE_KQUEUE)
#  define ASIO_HAS_KQUEUE 1
# endif // !defined(ASIO_DISABLE_KQUEUE)
#endif // (defined(__MACH__) && defined(__APPLE__))
       //   || defined(__FreeBSD__)
       //   || defined(__NetBSD__)
       //   || defined(__OpenBSD__)

// Solaris: /dev/poll.
#if defined(__sun)
# if !defined(ASIO_DISABLE_DEV_POLL)
#  define ASIO_HAS_DEV_POLL 1
# endif // !defined(ASIO_DISABLE_DEV_POLL)
#endif // defined(__sun)

// Serial ports.
#if defined(ASIO_HAS_IOCP) \
   || !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
# if !defined(ASIO_DISABLE_SERIAL_PORT)
#  define ASIO_HAS_SERIAL_PORT 1
# endif // !defined(ASIO_DISABLE_SERIAL_PORT)
#endif // defined(ASIO_HAS_IOCP)
       //   || !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

// Windows: stream handles.
#if !defined(ASIO_DISABLE_WINDOWS_STREAM_HANDLE)
# if defined(ASIO_HAS_IOCP)
#  define ASIO_HAS_WINDOWS_STREAM_HANDLE 1
# endif // defined(ASIO_HAS_IOCP)
#endif // !defined(ASIO_DISABLE_WINDOWS_STREAM_HANDLE)

// Windows: random access handles.
#if !defined(ASIO_DISABLE_WINDOWS_RANDOM_ACCESS_HANDLE)
# if defined(ASIO_HAS_IOCP)
#  define ASIO_HAS_WINDOWS_RANDOM_ACCESS_HANDLE 1
# endif // defined(ASIO_HAS_IOCP)
#endif // !defined(ASIO_DISABLE_WINDOWS_RANDOM_ACCESS_HANDLE)

// Windows: OVERLAPPED wrapper.
#if !defined(ASIO_DISABLE_WINDOWS_OVERLAPPED_PTR)
# if defined(ASIO_HAS_IOCP)
#  define ASIO_HAS_WINDOWS_OVERLAPPED_PTR 1
# endif // defined(ASIO_HAS_IOCP)
#endif // !defined(ASIO_DISABLE_WINDOWS_OVERLAPPED_PTR)

// POSIX: stream-oriented file descriptors.
#if !defined(ASIO_DISABLE_POSIX_STREAM_DESCRIPTOR)
# if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
#  define ASIO_HAS_POSIX_STREAM_DESCRIPTOR 1
# endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
#endif // !defined(ASIO_DISABLE_POSIX_STREAM_DESCRIPTOR)

// UNIX domain sockets.
#if !defined(ASIO_DISABLE_LOCAL_SOCKETS)
# if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
#  define ASIO_HAS_LOCAL_SOCKETS 1
# endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
#endif // !defined(ASIO_DISABLE_LOCAL_SOCKETS)

#endif // ASIO_DETAIL_CONFIG_HPP
