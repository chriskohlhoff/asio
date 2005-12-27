//
// bind.hpp
// ~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_BIND_HPP
#define ASIO_DETAIL_BIND_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/handler_alloc_hook.hpp"

// Custom bind handlers so that allocation hooks are correctly forwarded.

namespace asio {
namespace detail {

template <typename Handler, typename Arg1>
class binder1;

template <typename Handler, typename Arg1, typename Arg2>
class binder2;

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
class binder3;

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
class binder4;

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
class binder5;

} // namespace detail
} // namespace asio

namespace asio {

template <typename Handler, typename Arg1>
class handler_alloc_hook<
  asio::detail::binder1<Handler, Arg1> >;

template <typename Handler, typename Arg1, typename Arg2>
class handler_alloc_hook<
  asio::detail::binder2<Handler, Arg1, Arg2> >;

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
class handler_alloc_hook<
  asio::detail::binder3<Handler, Arg1, Arg2, Arg3> >;

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
class handler_alloc_hook<
  asio::detail::binder4<Handler, Arg1, Arg2, Arg3, Arg4> >;

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
class handler_alloc_hook<
  asio::detail::binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5> >;

} // namespace asio

namespace asio {
namespace detail {

template <typename Handler, typename Arg1>
class binder1
{
public:
  binder1(Handler handler, Arg1 arg1)
    : handler_(handler),
      arg1_(arg1)
  {
  }

  void operator()()
  {
    handler_(arg1_);
  }

  void operator()() const
  {
    handler_(arg1_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
  friend class asio::handler_alloc_hook<
    binder1<Handler, Arg1> >;
};

template <typename Handler, typename Arg1>
binder1<Handler, Arg1> bind_handler(Handler handler, Arg1 arg1)
{
  return binder1<Handler, Arg1>(handler, arg1);
}

template <typename Handler, typename Arg1, typename Arg2>
class binder2
{
public:
  binder2(Handler handler, Arg1 arg1, Arg2 arg2)
    : handler_(handler),
      arg1_(arg1),
      arg2_(arg2)
  {
  }

  void operator()()
  {
    handler_(arg1_, arg2_);
  }

  void operator()() const
  {
    handler_(arg1_, arg2_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  friend class asio::handler_alloc_hook<
    binder2<Handler, Arg1, Arg2> >;
};

template <typename Handler, typename Arg1, typename Arg2>
binder2<Handler, Arg1, Arg2> bind_handler(Handler handler, Arg1 arg1,
    Arg2 arg2)
{
  return binder2<Handler, Arg1, Arg2>(handler, arg1, arg2);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
class binder3
{
public:
  binder3(Handler handler, Arg1 arg1, Arg2 arg2, Arg3 arg3)
    : handler_(handler),
      arg1_(arg1),
      arg2_(arg2),
      arg3_(arg3)
  {
  }

  void operator()()
  {
    handler_(arg1_, arg2_, arg3_);
  }

  void operator()() const
  {
    handler_(arg1_, arg2_, arg3_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;
  friend class asio::handler_alloc_hook<
    binder3<Handler, Arg1, Arg2, Arg3> >;
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
binder3<Handler, Arg1, Arg2, Arg3> bind_handler(Handler handler, Arg1 arg1,
    Arg2 arg2, Arg3 arg3)
{
  return binder3<Handler, Arg1, Arg2, Arg3>(handler, arg1, arg2, arg3);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
class binder4
{
public:
  binder4(Handler handler, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
    : handler_(handler),
      arg1_(arg1),
      arg2_(arg2),
      arg3_(arg3),
      arg4_(arg4)
  {
  }

  void operator()()
  {
    handler_(arg1_, arg2_, arg3_, arg4_);
  }

  void operator()() const
  {
    handler_(arg1_, arg2_, arg3_, arg4_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;
  Arg4 arg4_;
  friend class asio::handler_alloc_hook<
    binder4<Handler, Arg1, Arg2, Arg3, Arg4> >;
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
binder4<Handler, Arg1, Arg2, Arg3, Arg4> bind_handler(Handler handler,
    Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  return binder4<Handler, Arg1, Arg2, Arg3, Arg4>(handler, arg1, arg2, arg3,
      arg4);
}

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
class binder5
{
public:
  binder5(Handler handler, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4,
      Arg5 arg5)
    : handler_(handler),
      arg1_(arg1),
      arg2_(arg2),
      arg3_(arg3),
      arg4_(arg4),
      arg5_(arg5)
  {
  }

  void operator()()
  {
    handler_(arg1_, arg2_, arg3_, arg4_, arg5_);
  }

  void operator()() const
  {
    handler_(arg1_, arg2_, arg3_, arg4_, arg5_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;
  Arg4 arg4_;
  Arg5 arg5_;
  friend class asio::handler_alloc_hook<
    binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5> >;
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5> bind_handler(Handler handler,
    Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  return binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>(handler, arg1, arg2,
      arg3, arg4, arg5);
}

} // namespace detail
} // namespace asio

template <typename Handler, typename Arg1>
class asio::handler_alloc_hook<
  asio::detail::binder1<Handler, Arg1> >
{
public:
  typedef asio::detail::binder1<Handler, Arg1> handler_type;

  template <typename Allocator>
  static typename Allocator::pointer allocate(handler_type& handler,
      Allocator& allocator, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::allocate(
        handler.handler_, allocator, count);
  }

  template <typename Allocator>
  static void deallocate(handler_type& handler, Allocator& allocator,
      typename Allocator::pointer pointer, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::deallocate(
        handler.handler_, allocator, pointer, count);
  }
};

template <typename Handler, typename Arg1, typename Arg2>
class asio::handler_alloc_hook<
  asio::detail::binder2<Handler, Arg1, Arg2> >
{
public:
  typedef asio::detail::binder2<Handler, Arg1, Arg2> handler_type;

  template <typename Allocator>
  static typename Allocator::pointer allocate(handler_type& handler,
      Allocator& allocator, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::allocate(
        handler.handler_, allocator, count);
  }

  template <typename Allocator>
  static void deallocate(handler_type& handler, Allocator& allocator,
      typename Allocator::pointer pointer, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::deallocate(
        handler.handler_, allocator, pointer, count);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
class asio::handler_alloc_hook<
  asio::detail::binder3<Handler, Arg1, Arg2, Arg3> >
{
public:
  typedef asio::detail::binder3<Handler, Arg1, Arg2, Arg3> handler_type;

  template <typename Allocator>
  static typename Allocator::pointer allocate(handler_type& handler,
      Allocator& allocator, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::allocate(
        handler.handler_, allocator, count);
  }

  template <typename Allocator>
  static void deallocate(handler_type& handler, Allocator& allocator,
      typename Allocator::pointer pointer, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::deallocate(
        handler.handler_, allocator, pointer, count);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4>
class asio::handler_alloc_hook<
  asio::detail::binder4<Handler, Arg1, Arg2, Arg3, Arg4> >
{
public:
  typedef asio::detail::binder4<
    Handler, Arg1, Arg2, Arg3, Arg4> handler_type;

  template <typename Allocator>
  static typename Allocator::pointer allocate(handler_type& handler,
      Allocator& allocator, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::allocate(
        handler.handler_, allocator, count);
  }

  template <typename Allocator>
  static void deallocate(handler_type& handler, Allocator& allocator,
      typename Allocator::pointer pointer, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::deallocate(
        handler.handler_, allocator, pointer, count);
  }
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3,
    typename Arg4, typename Arg5>
class asio::handler_alloc_hook<
  asio::detail::binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5> >
{
public:
  typedef asio::detail::binder5<
    Handler, Arg1, Arg2, Arg3, Arg4, Arg5> handler_type;

  template <typename Allocator>
  static typename Allocator::pointer allocate(handler_type& handler,
      Allocator& allocator, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::allocate(
        handler.handler_, allocator, count);
  }

  template <typename Allocator>
  static void deallocate(handler_type& handler, Allocator& allocator,
      typename Allocator::pointer pointer, typename Allocator::size_type count)
  {
    return asio::handler_alloc_hook<Handler>::deallocate(
        handler.handler_, allocator, pointer, count);
  }
};

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_BIND_HPP
