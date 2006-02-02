//
// handler_alloc_helpers.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_HANDLER_ALLOC_HELPERS_HPP
#define ASIO_DETAIL_HANDLER_ALLOC_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/handler_alloc_hook.hpp"
#include "asio/detail/noncopyable.hpp"

namespace asio {
namespace detail {

// Traits for handler allocation.
template <typename Handler, typename Object, typename Void_Allocator>
struct handler_alloc_traits
{
  typedef Handler handler_type;
  typedef Void_Allocator void_allocator_type;
  typedef typename Void_Allocator::template rebind<Object>::other
    allocator_type;
  typedef typename allocator_type::value_type value_type;
  typedef typename allocator_type::pointer pointer_type;
};

template <typename Alloc_Traits>
class handler_ptr;

// Helper class to provide RAII on uninitialised handler memory.
template <typename Alloc_Traits>
class raw_handler_ptr
  : private noncopyable
{
public:
  typedef typename Alloc_Traits::handler_type handler_type;
  typedef typename Alloc_Traits::void_allocator_type void_allocator_type;
  typedef typename Alloc_Traits::allocator_type allocator_type;
  typedef typename Alloc_Traits::value_type value_type;
  typedef typename Alloc_Traits::pointer_type pointer_type;
  typedef handler_alloc_hook<handler_type> hook_type;

  // Constructor allocates the memory.
  raw_handler_ptr(handler_type& handler,
      const void_allocator_type& void_allocator)
    : handler_(handler),
      allocator_(void_allocator),
      pointer_(hook_type::allocate(handler_, allocator_, 1))
  {
  }

  // Destructor automatically deallocates memory, unless it has been stolen by
  // a handler_ptr object.
  ~raw_handler_ptr()
  {
    if (pointer_)
      hook_type::deallocate(handler_, allocator_, pointer_, 1);
  }

private:
  friend class handler_ptr<Alloc_Traits>;
  handler_type& handler_;
  allocator_type allocator_;
  pointer_type pointer_;
};

// Helper class to provide RAII on uninitialised handler memory.
template <typename Alloc_Traits>
class handler_ptr
  : private noncopyable
{
public:
  typedef typename Alloc_Traits::handler_type handler_type;
  typedef typename Alloc_Traits::void_allocator_type void_allocator_type;
  typedef typename Alloc_Traits::allocator_type allocator_type;
  typedef typename Alloc_Traits::value_type value_type;
  typedef typename Alloc_Traits::pointer_type pointer_type;
  typedef handler_alloc_hook<handler_type> hook_type;
  typedef raw_handler_ptr<Alloc_Traits> raw_ptr_type;

  // Take ownership of existing memory.
  handler_ptr(handler_type& handler,
      const void_allocator_type& void_allocator, pointer_type pointer)
    : handler_(handler),
      allocator_(void_allocator),
      pointer_(pointer)
  {
  }

  // Construct object in raw memory and take ownership if construction succeeds.
  handler_ptr(raw_ptr_type& raw_ptr)
    : handler_(raw_ptr.handler_),
      allocator_(raw_ptr.allocator_),
      pointer_(new (raw_ptr.pointer_) value_type)
  {
    raw_ptr.pointer_ = 0;
  }

  // Construct object in raw memory and take ownership if construction succeeds.
  template <typename Arg1>
  handler_ptr(raw_ptr_type& raw_ptr, Arg1& a1)
    : handler_(raw_ptr.handler_),
      allocator_(raw_ptr.allocator_),
      pointer_(new (raw_ptr.pointer_) value_type(a1))
  {
    raw_ptr.pointer_ = 0;
  }

  // Construct object in raw memory and take ownership if construction succeeds.
  template <typename Arg1, typename Arg2>
  handler_ptr(raw_ptr_type& raw_ptr, Arg1& a1, Arg2& a2)
    : handler_(raw_ptr.handler_),
      allocator_(raw_ptr.allocator_),
      pointer_(new (raw_ptr.pointer_) value_type(a1, a2))
  {
    raw_ptr.pointer_ = 0;
  }

  // Construct object in raw memory and take ownership if construction succeeds.
  template <typename Arg1, typename Arg2, typename Arg3>
  handler_ptr(raw_ptr_type& raw_ptr, Arg1& a1, Arg2& a2, Arg3& a3)
    : handler_(raw_ptr.handler_),
      allocator_(raw_ptr.allocator_),
      pointer_(new (raw_ptr.pointer_) value_type(a1, a2, a3))
  {
    raw_ptr.pointer_ = 0;
  }

  // Construct object in raw memory and take ownership if construction succeeds.
  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  handler_ptr(raw_ptr_type& raw_ptr, Arg1& a1, Arg2& a2, Arg3& a3, Arg4& a4)
    : handler_(raw_ptr.handler_),
      allocator_(raw_ptr.allocator_),
      pointer_(new (raw_ptr.pointer_) value_type(a1, a2, a3, a4))
  {
    raw_ptr.pointer_ = 0;
  }

  // Construct object in raw memory and take ownership if construction succeeds.
  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
      typename Arg5>
  handler_ptr(raw_ptr_type& raw_ptr, Arg1& a1, Arg2& a2, Arg3& a3, Arg4& a4,
      Arg5& a5)
    : handler_(raw_ptr.handler_),
      allocator_(raw_ptr.allocator_),
      pointer_(new (raw_ptr.pointer_) value_type(a1, a2, a3, a4, a5))
  {
    raw_ptr.pointer_ = 0;
  }

  // Construct object in raw memory and take ownership if construction succeeds.
  template <typename Arg1, typename Arg2, typename Arg3, typename Arg4,
      typename Arg5, typename Arg6>
  handler_ptr(raw_ptr_type& raw_ptr, Arg1& a1, Arg2& a2, Arg3& a3, Arg4& a4,
      Arg5& a5, Arg6& a6)
    : handler_(raw_ptr.handler_),
      allocator_(raw_ptr.allocator_),
      pointer_(new (raw_ptr.pointer_) value_type(a1, a2, a3, a4, a5, a6))
  {
    raw_ptr.pointer_ = 0;
  }

  // Destructor automatically deallocates memory, unless it has been released.
  ~handler_ptr()
  {
    reset();
  }

  // Get the memory.
  pointer_type get() const
  {
    return pointer_;
  }

  // Release ownership of the memory.
  pointer_type release()
  {
    pointer_type tmp = pointer_;
    pointer_ = 0;
    return tmp;
  }

  // Explicitly destroy and deallocate the memory.
  void reset()
  {
    if (pointer_)
    {
      allocator_.destroy(pointer_);
      hook_type::deallocate(handler_, allocator_, pointer_, 1);
      pointer_ = 0;
    }
  }

private:
  handler_type& handler_;
  allocator_type allocator_;
  pointer_type pointer_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HANDLER_ALLOC_HELPERS_HPP
