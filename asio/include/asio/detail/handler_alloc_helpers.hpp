//
// detail/handler_alloc_helpers.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_ALLOC_HELPERS_HPP
#define ASIO_DETAIL_HANDLER_ALLOC_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <boost/detail/workaround.hpp>
#include <boost/limits.hpp>
#include <boost/utility/addressof.hpp>
#include "asio/detail/noncopyable.hpp"
#include "asio/handler_alloc_hook.hpp"

#include "asio/detail/push_options.hpp"

// Calls to asio_handler_allocate and asio_handler_deallocate must be made from
// a namespace that does not contain any overloads of these functions. The
// asio_handler_alloc_helpers namespace is defined here for that purpose.
namespace asio_handler_alloc_helpers {

template <typename Handler>
inline void* allocate(std::size_t s, Handler& h)
{
#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564)) \
  || BOOST_WORKAROUND(__GNUC__, < 3)
  return ::operator new(s);
#else
  using namespace asio;
  return asio_handler_allocate(s, boost::addressof(h));
#endif
}

template <typename Handler>
inline void deallocate(void* p, std::size_t s, Handler& h)
{
#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564)) \
  || BOOST_WORKAROUND(__GNUC__, < 3)
  ::operator delete(p);
#else
  using namespace asio;
  asio_handler_deallocate(p, s, boost::addressof(h));
#endif
}

} // namespace asio_handler_alloc_helpers

namespace asio {
namespace detail {

// The default allocator simply forwards to the old-style allocation hook.
template <typename T, typename Context>
class default_handler_allocator
{
public:
  typedef T value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  template <typename U>
  struct rebind
  {
    typedef default_handler_allocator<U, Context> other;
  };

  explicit default_handler_allocator(Context* context)
    : context_(context)
  {
  }

  template <typename U>
  default_handler_allocator(const default_handler_allocator<U, Context>& other)
    : context_(other.context_)
  {
  }

  static pointer address(reference r)
  {
    return &r;
  }

  static const_pointer address(const_reference r)
  {
    return &r;
  }

  static size_type max_size()
  {
    return (std::numeric_limits<size_type>::max)();
  }

  static void construct(const pointer p, const value_type& v)
  {
    new (p) T(v);
  }

  static void destroy(const pointer p)
  {
    p->~T();
  }

  bool operator==(const default_handler_allocator& other) const
  {
    return context_ == other.context_;
  }

  bool operator!=(const default_handler_allocator& other) const
  {
    return context_ != other.context_;
  }

  pointer allocate(size_type n, const void* = 0)
  {
    return static_cast<pointer>(
        asio_handler_alloc_helpers::allocate(n * sizeof(T), *context_));
  }

  void deallocate(pointer p, size_type n)
  {
    return asio_handler_alloc_helpers::deallocate(p, n * sizeof(T), *context_);
  }

//private:
  Context* context_;
};

// The default allocator specialised for void.
template <typename Context>
class default_handler_allocator<void, Context>
{
public:
  typedef void value_type;
  typedef void* pointer;
  typedef const void* const_pointer;

  template <typename U>
  struct rebind
  {
    typedef default_handler_allocator<U, Context> other;
  };

  explicit default_handler_allocator(Context* context)
    : context_(context)
  {
  }

  template <typename U>
  default_handler_allocator(const default_handler_allocator<U, Context>& other)
    : context_(other.context_)
  {
  }

  bool operator==(const default_handler_allocator& other) const
  {
    return context_ == other.context_;
  }

  bool operator!=(const default_handler_allocator& other) const
  {
    return context_ != other.context_;
  }

//private:
  Context* context_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HANDLER_ALLOC_HELPERS_HPP
