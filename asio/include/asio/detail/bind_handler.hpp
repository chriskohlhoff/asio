//
// detail/bind_handler.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_BIND_HANDLER_HPP
#define ASIO_DETAIL_BIND_HANDLER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associated_allocator.hpp"
#include "asio/associated_executor.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_cont_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Handler, typename Arg1>
class binder1
{
public:
  binder1(const Handler& handler, const Arg1& arg1)
    : handler_(handler),
      arg1_(arg1)
  {
  }

  binder1(Handler& handler, const Arg1& arg1)
    : handler_(ASIO_MOVE_CAST(Handler)(handler)),
      arg1_(arg1)
  {
  }

  void operator()()
  {
    handler_(static_cast<const Arg1&>(arg1_));
  }

  void operator()() const
  {
    handler_(arg1_);
  }

//private:
  Handler handler_;
  Arg1 arg1_;
};

template <typename Handler, typename Arg1>
inline void* asio_handler_allocate(std::size_t size,
    binder1<Handler, Arg1>* this_handler)
{
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->handler_);
}

template <typename Handler, typename Arg1>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
    binder1<Handler, Arg1>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->handler_);
}

template <typename Handler, typename Arg1>
inline bool asio_handler_is_continuation(
    binder1<Handler, Arg1>* this_handler)
{
  return asio_handler_cont_helpers::is_continuation(
      this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1>
inline void asio_handler_invoke(Function& function,
    binder1<Handler, Arg1>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1>
inline void asio_handler_invoke(const Function& function,
    binder1<Handler, Arg1>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Handler, typename Arg1>
inline binder1<Handler, Arg1> bind_handler(Handler handler,
    const Arg1& arg1)
{
  return binder1<Handler, Arg1>(handler, arg1);
}

template <typename Handler, typename Arg1, typename Arg2>
class binder2
{
public:
  binder2(const Handler& handler, const Arg1& arg1, const Arg2& arg2)
    : handler_(handler),
      arg1_(arg1),
      arg2_(arg2)
  {
  }

  binder2(Handler& handler, const Arg1& arg1, const Arg2& arg2)
    : handler_(ASIO_MOVE_CAST(Handler)(handler)),
      arg1_(arg1),
      arg2_(arg2)
  {
  }

  void operator()()
  {
    handler_(static_cast<const Arg1&>(arg1_),
        static_cast<const Arg2&>(arg2_));
  }

  void operator()() const
  {
    handler_(arg1_, arg2_);
  }

//private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
};

template <typename Handler, typename Arg1, typename Arg2>
inline void* asio_handler_allocate(std::size_t size,
    binder2<Handler, Arg1, Arg2>* this_handler)
{
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
    binder2<Handler, Arg1, Arg2>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2>
inline bool asio_handler_is_continuation(
    binder2<Handler, Arg1, Arg2>* this_handler)
{
  return asio_handler_cont_helpers::is_continuation(
      this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1, typename Arg2>
inline void asio_handler_invoke(Function& function,
    binder2<Handler, Arg1, Arg2>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1, typename Arg2>
inline void asio_handler_invoke(const Function& function,
    binder2<Handler, Arg1, Arg2>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2>
inline binder2<Handler, Arg1, Arg2> bind_handler(Handler handler,
    const Arg1& arg1, const Arg2& arg2)
{
  return binder2<Handler, Arg1, Arg2>(handler, arg1, arg2);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
class binder3
{
public:
  binder3(const Handler& handler, const Arg1& arg1, const Arg2& arg2,
      const Arg3& arg3)
    : handler_(handler),
      arg1_(arg1),
      arg2_(arg2),
      arg3_(arg3)
  {
  }

  binder3(Handler& handler, const Arg1& arg1, const Arg2& arg2,
      const Arg3& arg3)
    : handler_(ASIO_MOVE_CAST(Handler)(handler)),
      arg1_(arg1),
      arg2_(arg2),
      arg3_(arg3)
  {
  }

  void operator()()
  {
    handler_(static_cast<const Arg1&>(arg1_),
        static_cast<const Arg2&>(arg2_),
        static_cast<const Arg3&>(arg3_));
  }

  void operator()() const
  {
    handler_(arg1_, arg2_, arg3_);
  }

//private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
inline void* asio_handler_allocate(std::size_t size,
    binder3<Handler, Arg1, Arg2, Arg3>* this_handler)
{
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
    binder3<Handler, Arg1, Arg2, Arg3>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
inline bool asio_handler_is_continuation(
    binder3<Handler, Arg1, Arg2, Arg3>* this_handler)
{
  return asio_handler_cont_helpers::is_continuation(
      this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1, typename Arg2,
    typename Arg3>
inline void asio_handler_invoke(Function& function,
    binder3<Handler, Arg1, Arg2, Arg3>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1, typename Arg2,
    typename Arg3>
inline void asio_handler_invoke(const Function& function,
    binder3<Handler, Arg1, Arg2, Arg3>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
inline binder3<Handler, Arg1, Arg2, Arg3> bind_handler(Handler handler,
    const Arg1& arg1, const Arg2& arg2, const Arg3& arg3)
{
  return binder3<Handler, Arg1, Arg2, Arg3>(handler, arg1, arg2, arg3);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
class binder4
{
public:
  binder4(const Handler& handler, const Arg1& arg1, const Arg2& arg2,
      const Arg3& arg3, const Arg4& arg4)
    : handler_(handler),
      arg1_(arg1),
      arg2_(arg2),
      arg3_(arg3),
      arg4_(arg4)
  {
  }

  binder4(Handler& handler, const Arg1& arg1, const Arg2& arg2,
      const Arg3& arg3, const Arg4& arg4)
    : handler_(ASIO_MOVE_CAST(Handler)(handler)),
      arg1_(arg1),
      arg2_(arg2),
      arg3_(arg3),
      arg4_(arg4)
  {
  }

  void operator()()
  {
    handler_(static_cast<const Arg1&>(arg1_),
        static_cast<const Arg2&>(arg2_),
        static_cast<const Arg3&>(arg3_),
        static_cast<const Arg4&>(arg4_));
  }

  void operator()() const
  {
    handler_(arg1_, arg2_, arg3_, arg4_);
  }

//private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;
  Arg4 arg4_;
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
inline void* asio_handler_allocate(std::size_t size,
    binder4<Handler, Arg1, Arg2, Arg3, Arg4>* this_handler)
{
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
    binder4<Handler, Arg1, Arg2, Arg3, Arg4>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
inline bool asio_handler_is_continuation(
    binder4<Handler, Arg1, Arg2, Arg3, Arg4>* this_handler)
{
  return asio_handler_cont_helpers::is_continuation(
      this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1, typename Arg2,
    typename Arg3, typename Arg4>
inline void asio_handler_invoke(Function& function,
    binder4<Handler, Arg1, Arg2, Arg3, Arg4>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1, typename Arg2,
    typename Arg3, typename Arg4>
inline void asio_handler_invoke(const Function& function,
    binder4<Handler, Arg1, Arg2, Arg3, Arg4>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
inline binder4<Handler, Arg1, Arg2, Arg3, Arg4> bind_handler(
    Handler handler, const Arg1& arg1, const Arg2& arg2,
    const Arg3& arg3, const Arg4& arg4)
{
  return binder4<Handler, Arg1, Arg2, Arg3, Arg4>(handler, arg1, arg2, arg3,
      arg4);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
class binder5
{
public:
  binder5(const Handler& handler, const Arg1& arg1, const Arg2& arg2,
      const Arg3& arg3, const Arg4& arg4, const Arg5& arg5)
    : handler_(handler),
      arg1_(arg1),
      arg2_(arg2),
      arg3_(arg3),
      arg4_(arg4),
      arg5_(arg5)
  {
  }

  binder5(Handler& handler, const Arg1& arg1, const Arg2& arg2,
      const Arg3& arg3, const Arg4& arg4, const Arg5& arg5)
    : handler_(ASIO_MOVE_CAST(Handler)(handler)),
      arg1_(arg1),
      arg2_(arg2),
      arg3_(arg3),
      arg4_(arg4),
      arg5_(arg5)
  {
  }

  void operator()()
  {
    handler_(static_cast<const Arg1&>(arg1_),
        static_cast<const Arg2&>(arg2_),
        static_cast<const Arg3&>(arg3_),
        static_cast<const Arg4&>(arg4_),
        static_cast<const Arg5&>(arg5_));
  }

  void operator()() const
  {
    handler_(arg1_, arg2_, arg3_, arg4_, arg5_);
  }

//private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;
  Arg4 arg4_;
  Arg5 arg5_;
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
inline void* asio_handler_allocate(std::size_t size,
    binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>* this_handler)
{
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
    binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
inline bool asio_handler_is_continuation(
    binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>* this_handler)
{
  return asio_handler_cont_helpers::is_continuation(
      this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1, typename Arg2,
    typename Arg3, typename Arg4, typename Arg5>
inline void asio_handler_invoke(Function& function,
    binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Function, typename Handler, typename Arg1, typename Arg2,
    typename Arg3, typename Arg4, typename Arg5>
inline void asio_handler_invoke(const Function& function,
    binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
inline binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5> bind_handler(
    Handler handler, const Arg1& arg1, const Arg2& arg2,
    const Arg3& arg3, const Arg4& arg4, const Arg5& arg5)
{
  return binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>(handler, arg1, arg2,
      arg3, arg4, arg5);
}

} // namespace detail

template <typename Handler, typename Arg1, typename Allocator>
struct associated_allocator<detail::binder1<Handler, Arg1>, Allocator>
{
  typedef typename associated_allocator<Handler, Allocator>::type type;

  static type get(const detail::binder1<Handler, Arg1>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<Handler, Allocator>::get(h.handler_, a);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Allocator>
struct associated_allocator<detail::binder2<Handler, Arg1, Arg2>, Allocator>
{
  typedef typename associated_allocator<Handler, Allocator>::type type;

  static type get(const detail::binder2<Handler, Arg1, Arg2>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<Handler, Allocator>::get(h.handler_, a);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Allocator>
struct associated_allocator<detail::binder3<Handler, Arg1, Arg2, Arg3>,
    Allocator>
{
  typedef typename associated_allocator<Handler, Allocator>::type type;

  static type get(const detail::binder3<Handler, Arg1, Arg2, Arg3>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<Handler, Allocator>::get(h.handler_, a);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Allocator>
struct associated_allocator<detail::binder4<Handler, Arg1, Arg2, Arg3, Arg4>,
    Allocator>
{
  typedef typename associated_allocator<Handler, Allocator>::type type;

  static type get(const detail::binder4<Handler, Arg1, Arg2, Arg3, Arg4>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<Handler, Allocator>::get(h.handler_, a);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Allocator>
struct associated_allocator<
    detail::binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>, Allocator>
{
  typedef typename associated_allocator<Handler, Allocator>::type type;

  static type get(
      const detail::binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>& h,
      const Allocator& a = Allocator()) ASIO_NOEXCEPT
  {
    return associated_allocator<Handler, Allocator>::get(h.handler_, a);
  }
};

template <typename Handler, typename Arg1, typename Executor>
struct associated_executor<detail::binder1<Handler, Arg1>, Executor>
{
  typedef typename associated_executor<Handler, Executor>::type type;

  static type get(const detail::binder1<Handler, Arg1>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<Handler, Executor>::get(h.handler_, ex);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Executor>
struct associated_executor<detail::binder2<Handler, Arg1, Arg2>, Executor>
{
  typedef typename associated_executor<Handler, Executor>::type type;

  static type get(const detail::binder2<Handler, Arg1, Arg2>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<Handler, Executor>::get(h.handler_, ex);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Executor>
struct associated_executor<detail::binder3<Handler, Arg1, Arg2, Arg3>,
    Executor>
{
  typedef typename associated_executor<Handler, Executor>::type type;

  static type get(const detail::binder3<Handler, Arg1, Arg2, Arg3>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<Handler, Executor>::get(h.handler_, ex);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Executor>
struct associated_executor<detail::binder4<Handler, Arg1, Arg2, Arg3, Arg4>,
    Executor>
{
  typedef typename associated_executor<Handler, Executor>::type type;

  static type get(const detail::binder4<Handler, Arg1, Arg2, Arg3, Arg4>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<Handler, Executor>::get(h.handler_, ex);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5, typename Executor>
struct associated_executor<
    detail::binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>, Executor>
{
  typedef typename associated_executor<Handler, Executor>::type type;

  static type get(
      const detail::binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>& h,
      const Executor& ex = Executor()) ASIO_NOEXCEPT
  {
    return associated_executor<Handler, Executor>::get(h.handler_, ex);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_BIND_HANDLER_HPP
