//
// detail/scheduler_allocator.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SCHEDULER_ALLOCATOR_HPP
#define ASIO_DETAIL_SCHEDULER_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/scheduler_thread_info.hpp"
#include "asio/unspecified_allocator.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T = void>
class scheduler_allocator
{
public:
  template <typename U>
  struct rebind
  {
    typedef scheduler_allocator<U> other;
  };

  scheduler_allocator()
  {
  }

  template <typename U>
  scheduler_allocator(const scheduler_allocator<U>&)
  {
  }

  T* allocate(std::size_t n)
  {
    typedef scheduler_thread_info thread_info;
    typedef call_stack<scheduler, thread_info> call_stack;
    void* p = thread_info::allocate(call_stack::top(), sizeof(T) * n);
    return static_cast<T*>(p);
  }

  void deallocate(T* p, std::size_t n)
  {
    typedef scheduler_thread_info thread_info;
    typedef call_stack<scheduler, thread_info> call_stack;
    thread_info::deallocate(call_stack::top(), p, sizeof(T) * n);
  }
};

template <>
class scheduler_allocator<void>
{
public:
  template <typename U>
  struct rebind
  {
    typedef scheduler_allocator<U> other;
  };

  scheduler_allocator()
  {
  }

  template <typename U>
  scheduler_allocator(const scheduler_allocator<U>&)
  {
  }
};

template <typename Allocator>
struct get_scheduler_allocator
{
  typedef Allocator type;
  static type get(const Allocator& a) { return a; }
};

template <typename T>
struct get_scheduler_allocator<std::allocator<T> >
{
  typedef scheduler_allocator<T> type;
  static type get(const std::allocator<T>&) { return type(); }
};

template <typename T>
struct get_scheduler_allocator<unspecified_allocator<T> >
{
  typedef scheduler_allocator<T> type;
  static type get(const unspecified_allocator<T>&) { return type(); }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SCHEDULER_ALLOCATOR_HPP
