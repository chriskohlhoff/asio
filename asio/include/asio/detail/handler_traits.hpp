//
// detail/handler_traits.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_TRAITS_HPP
#define ASIO_DETAIL_HANDLER_TRAITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <boost/utility/addressof.hpp>
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Type trait for determining whether a handler has an invoker_type typedef.

char (&invoker_type_test(...))[2];

template <typename T> char invoker_type_test(
    T*, typename T::invoker_type* = 0);

template <typename T>
struct has_invoker_type
{
  enum { value = (sizeof((invoker_type_test)(static_cast<T*>(0))) == 1) };
};

// Type trait for determining whether a handler has an allocator_type typedef.

char (&allocator_type_test(...))[2];

template <typename T> char allocator_type_test(
    T*, typename T::allocator_type* = 0);

template <typename T>
struct has_allocator_type
{
  enum { value = (sizeof((allocator_type_test)(static_cast<T*>(0))) == 1) };
};

// Traits base class for selectively forwarding the invoker typedef and
// accessor function to the handler class.

template <typename Handler, bool HasInvoker = has_invoker_type<Handler>::value>
struct handler_traits_invoker;

template <typename Handler>
struct handler_traits_invoker<Handler, true>
{
  typedef typename Handler::invoker_type invoker_type;

  static invoker_type get_invoker(Handler& handler)
  {
    return handler.get_invoker();
  }
};

template <typename Handler>
struct handler_traits_invoker<Handler, false>
{
  typedef default_handler_invoker<Handler> invoker_type;

  static invoker_type get_invoker(Handler& handler)
  {
    return invoker_type(boost::addressof(handler));
  }
};

// Traits base class for selectively forwarding the allocator typedef and
// accessor function to the handler class.

template <typename Handler, bool HasAllocator = has_allocator_type<Handler>::value>
struct handler_traits_allocator;

template <typename Handler>
struct handler_traits_allocator<Handler, true>
{
  typedef typename Handler::allocator_type allocator_type;

  static allocator_type get_allocator(Handler& handler)
  {
    return handler.get_allocator();
  }
};

template <typename Handler>
struct handler_traits_allocator<Handler, false>
{
  typedef default_handler_allocator<void, Handler> allocator_type;

  static allocator_type get_allocator(Handler& handler)
  {
    return allocator_type(boost::addressof(handler));
  }
};

// The default handler traits.

template <typename Handler>
struct handler_traits :
  handler_traits_invoker<Handler>,
  handler_traits_allocator<Handler>
{
};

} // namespace detail
} // namespace asio

#define ASIO_DEFINE_HANDLER_PTR(op) \
  struct ptr \
  { \
    typedef asio::handler_traits<Handler> traits_type; \
    typedef typename traits_type::allocator_type any_allocator_type; \
    typedef typename any_allocator_type::template rebind< \
        op>::other allocator_type; \
    Handler* h; \
    void* v; \
    op* p; \
    static op* allocate(Handler& handler) \
    { \
      allocator_type allocator = traits_type::get_allocator(handler); \
      return allocator.allocate(1); \
    } \
    ~ptr() \
    { \
      reset(); \
    } \
    void reset() \
    { \
      if (p) \
      { \
        p->~op(); \
        p = 0; \
      } \
      if (v) \
      { \
        allocator_type allocator = traits_type::get_allocator(*h); \
        allocator.deallocate(static_cast<op*>(v), 1); \
        v = 0; \
      } \
    } \
  } \
  /**/

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HANDLER_TRAITS_HPP
