//
// detail/task_io_service_allocator.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TASK_IO_SERVICE_ALLOCATOR_HPP
#define ASIO_DETAIL_TASK_IO_SERVICE_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/task_io_service_thread_info.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T = void>
class task_io_service_allocator
{
public:
  template <typename U>
  struct rebind
  {
    typedef task_io_service_allocator<U> type;
  };

  task_io_service_allocator()
  {
  }

  template <typename U>
  task_io_service_allocator(const task_io_service_allocator<U>&)
  {
  }

  T* allocate(std::size_t n)
  {
    typedef task_io_service_thread_info thread_info;
    typedef call_stack<task_io_service, thread_info> call_stack;
    void* p = thread_info::allocate(call_stack::top(), sizeof(T) * n);
    return static_cast<T*>(p);
  }

  void deallocate(T* p, std::size_t n)
  {
    typedef task_io_service_thread_info thread_info;
    typedef call_stack<task_io_service, thread_info> call_stack;
    thread_info::deallocate(call_stack::top(), p, sizeof(T) * n);
  }
};

template <>
class task_io_service_allocator<void>
{
public:
  template <typename U>
  struct rebind
  {
    typedef task_io_service_allocator<U> type;
  };

  task_io_service_allocator()
  {
  }

  template <typename U>
  task_io_service_allocator(const task_io_service_allocator<U>&)
  {
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TASK_IO_SERVICE_ALLOCATOR_HPP
