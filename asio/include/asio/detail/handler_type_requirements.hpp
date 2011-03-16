//
// detail/handler_type_requirements.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_TYPE_REQUIREMENTS_HPP
#define ASIO_DETAIL_HANDLER_TYPE_REQUIREMENTS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

namespace asio {
namespace detail {

// Older versions of gcc have difficulty compiling the sizeof expressions where
// we test the handler type requirements. We'll disable checking of handler type
// requirements for those compilers, but otherwise enable it by default.
#if !defined(ASIO_DISABLE_HANDLER_TYPE_REQUIREMENTS)
# if !defined(__GNUC__) || (__GNUC__ >= 4)
#  define ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS 1
# endif // !defined(__GNUC__) || (__GNUC__ >= 4)
#endif // !defined(ASIO_DISABLE_HANDLER_TYPE_REQUIREMENTS)

// With C++0x we can use a combination of enhanced SFINAE and static_assert to
// generate better template error messages. As this technique is not yet widely
// portable, we'll only enable it for tested compilers.
#if !defined(ASIO_DISABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT)
# if defined(__GNUC__)
#  if ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 5)) || (__GNUC__ > 4)
#   if defined(__GXX_EXPERIMENTAL_CXX0X__)
#    define ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT 1
#   endif // defined(__GXX_EXPERIMENTAL_CXX0X__)
#  endif // ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 5)) || (__GNUC__ > 4)
# endif // defined(__GNUC__)
# if defined(BOOST_MSVC)
#  if (_MSC_VER >= 1600)
#   define ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT 1
#  endif // (_MSC_VER >= 1600)
# endif // defined(BOOST_MSVC)
#endif // !defined(ASIO_DISABLE_HANDLER_TYPE_REQUIREMENTS)

#if defined(ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS)

# if defined(ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT)

template <typename Handler>
auto zero_arg_handler_test(Handler* h)
  -> decltype(
    sizeof(Handler(*static_cast<const Handler*>(h))),
    ((*h)()),
    char(0));

char (&zero_arg_handler_test(...))[2];

template <typename Handler, typename Arg1>
auto one_arg_handler_test(Handler* h, Arg1* a1)
  -> decltype(
    sizeof(Handler(*static_cast<const Handler*>(h))),
    ((*h)(*a1)),
    char(0));

char (&one_arg_handler_test(...))[2];

template <typename Handler, typename Arg1, typename Arg2>
auto two_arg_handler_test(Handler* h, Arg1* a1, Arg2* a2)
  -> decltype(
    sizeof(Handler(*static_cast<const Handler*>(h))),
    ((*h)(*a1, *a2)),
    char(0));

char (&two_arg_handler_test(...))[2];

#  define ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT(expr, msg) \
     static_assert(expr, msg);

# else // defined(ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT)

#  define ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT(expr, msg)

# endif // defined(ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT)

template <typename T> T& lvref();

template <int>
struct handler_type_requirements
{
};

#define ASIO_COMPLETION_HANDLER_CHECK( \
    handler_type, handler) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::zero_arg_handler_test( \
          static_cast<handler_type*>(0))) == 1, \
      "CompletionHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler(), \
        char(0))>

#define ASIO_READ_HANDLER_CHECK( \
    handler_type, handler) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::two_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0), \
          static_cast<const std::size_t*>(0))) == 1, \
      "ReadHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>(), \
          asio::detail::lvref<const std::size_t>()), \
        char(0))>

#define ASIO_WRITE_HANDLER_CHECK( \
    handler_type, handler) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::two_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0), \
          static_cast<const std::size_t*>(0))) == 1, \
      "WriteHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>(), \
          asio::detail::lvref<const std::size_t>()), \
        char(0))>

#define ASIO_ACCEPT_HANDLER_CHECK( \
    handler_type, handler) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::one_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0))) == 1, \
      "AcceptHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>()), \
        char(0))>

#define ASIO_CONNECT_HANDLER_CHECK( \
    handler_type, handler) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::one_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0))) == 1, \
      "ConnectHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>()), \
        char(0))>

#define ASIO_COMPOSED_CONNECT_HANDLER_CHECK( \
    handler_type, handler, iter_type) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::two_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0), \
          static_cast<const iter_type*>(0))) == 1, \
      "ComposedConnectHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>(), \
          asio::detail::lvref<const iter_type>()), \
        char(0))>

#define ASIO_RESOLVE_HANDLER_CHECK( \
    handler_type, handler, iter_type) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::two_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0), \
          static_cast<const iter_type*>(0))) == 1, \
      "ResolveHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>(), \
          asio::detail::lvref<const iter_type>()), \
        char(0))>

#define ASIO_WAIT_HANDLER_CHECK( \
    handler_type, handler) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::one_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0))) == 1, \
      "WaitHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>()), \
        char(0))>

#define ASIO_SIGNAL_HANDLER_CHECK( \
    handler_type, handler) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::two_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0), \
          static_cast<const int*>(0))) == 1, \
      "SignalHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>(), \
          asio::detail::lvref<const int>()), \
        char(0))>

#define ASIO_HANDSHAKE_HANDLER_CHECK( \
    handler_type, handler) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::one_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0))) == 1, \
      "HandshakeHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>()), \
        char(0))>

#define ASIO_SHUTDOWN_HANDLER_CHECK( \
    handler_type, handler) \
  \
  ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
      sizeof(asio::detail::one_arg_handler_test( \
          static_cast<handler_type*>(0), \
          static_cast<const asio::error_code*>(0))) == 1, \
      "ShutdownHandler type requirements not met") \
  \
  typedef asio::detail::handler_type_requirements< \
      sizeof( \
        handler_type( \
          static_cast<const handler_type&>(handler))) + \
      sizeof( \
        handler( \
          asio::detail::lvref<const asio::error_code>()), \
        char(0))>

#else // !defined(ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS)

#define ASIO_COMPLETION_HANDLER_CHECK( \
    handler_type, handler) \
  typedef int

#define ASIO_READ_HANDLER_CHECK( \
    handler_type, handler) \
  typedef int

#define ASIO_WRITE_HANDLER_CHECK( \
    handler_type, handler) \
  typedef int

#define ASIO_ACCEPT_HANDLER_CHECK( \
    handler_type, handler) \
  typedef int

#define ASIO_CONNECT_HANDLER_CHECK( \
    handler_type, handler) \
  typedef int

#define ASIO_COMPOSED_CONNECT_HANDLER_CHECK( \
    handler_type, handler, iter_type) \
  typedef int

#define ASIO_RESOLVE_HANDLER_CHECK( \
    handler_type, handler, iter_type) \
  typedef int

#define ASIO_WAIT_HANDLER_CHECK( \
    handler_type, handler) \
  typedef int

#define ASIO_SIGNAL_HANDLER_CHECK( \
    handler_type, handler) \
  typedef int

#define ASIO_HANDSHAKE_HANDLER_CHECK( \
    handler_type, handler) \
  typedef int

#define ASIO_SHUTDOWN_HANDLER_CHECK( \
    handler_type, handler) \
  typedef int

#endif // !defined(ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS)

} // namespace detail
} // namespace asio

#endif // ASIO_DETAIL_HANDLER_TYPE_REQUIREMENTS_HPP
