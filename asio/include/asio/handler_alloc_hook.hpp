//
// handler_alloc_hook.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_HANDLER_ALLOC_HOOK_HPP
#define ASIO_HANDLER_ALLOC_HOOK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

namespace asio {

/// Allocation hook for handlers.
/**
 * Asynchronous operations may need to allocate temporary objects. Since
 * asynchronous operations have a handler function object, these temporary
 * objects can be said to be associated with the handler.
 *
 * Specialise this class template for your own handlers to provide custom
 * allocation for these temporary objects. The default implementation simply
 * forwards the calls to the supplied allocator object.
 *
 * @note All temporary objects associated with a handler will be deallocated
 * before the upcall to the handler is performed. This allows the same memory to
 * be reused for a subsequent asynchronous operation initiated by the handler.
 *
 * @par Example:
 * @code
 * class my_handler;
 *
 * template <>
 * class asio::handler_alloc_hook<my_handler>
 * {
 * public:
 *   template <typename Allocator>
 *   static typename Allocator::pointer allocate(
 *       Handler& handler, Allocator& allocator,
 *       typename Allocator::size_type count)
 *   {
 *     typedef typename Allocator::pointer pointer_type;
 *     typedef typename Allocator::value_type value_type;
 *     void* mem = ::operator new(sizeof(value_type) * count);
 *     return static_cast<pointer_type>(mem);
 *   }
 *
 *   template <typename Allocator>
 *   static void deallocate(Handler& handler,
 *       Allocator& allocator,
 *       typename Allocator::pointer pointer,
 *       typename Allocator::size_type count)
 *   {
 *     ::operator delete(pointer);
 *   }
 * };
 * @endcode
 */
template <typename Handler>
class handler_alloc_hook
{
public:
  /**
   * Handle a request to allocate some memory associated with a handler. The
   * default implementation is:
   * @code
   * return allocator.allocate(count);
   * @endcode
   *
   * @param handler A reference to the user handler object. May be used to
   * access pre-allocated memory that is associated with a handler object. Note
   * that this handler may be a copy of the original handler object passed to
   * the original function.
   *
   * @param allocator The allocator object associated with the demuxer. The
   * allocator has been rebound such that its value_type is the internal asio
   * type to be allocated.
   *
   * @param count The number of objects to be allocated.
   *
   * @throws std::bad_alloc Thrown if memory cannot be allocated.
   */
  template <typename Allocator>
  static typename Allocator::pointer allocate(Handler& handler,
    Allocator& allocator, typename Allocator::size_type count)
  {
    return allocator.allocate(count);
  }

  /**
   * Handle a request to deallocate some memory associated with a handler. The
   * default implementation is:
   * @code
   * allocator.deallocate(pointer, count);
   * @endcode
   *
   * @param handler A reference to the user handler object. May be used to
   * access pre-allocated memory that is associated with a handler object. Note
   * that this handler may be a copy of the original handler object passed to
   * the original function.
   *
   * @param allocator The allocator object associated with the demuxer. The
   * allocator has been rebound such that its value_type is the internal asio
   * type to be allocated.
   *
   * @param pointer A pointer to the memory to be deallocated.
   *
   * @param count The number of objects to be deallocated.
   */
  template <typename Allocator>
  static void deallocate(Handler& handler, Allocator& allocator,
      typename Allocator::pointer pointer, typename Allocator::size_type count)
  {
    allocator.deallocate(pointer, count);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_HANDLER_ALLOC_HOOK_HPP
