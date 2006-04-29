//
// bind.hpp
// ~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

#include "asio/detail/handler_alloc_helpers.hpp"

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

  friend void* asio_handler_allocate(std::size_t size,
      binder1<Handler, Arg1>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, &this_handler->handler_);
  }

  friend void asio_handler_deallocate(void* pointer, std::size_t size,
      binder1<Handler, Arg1>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, &this_handler->handler_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
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

  friend void* asio_handler_allocate(std::size_t size,
      binder2<Handler, Arg1, Arg2>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, &this_handler->handler_);
  }

  friend void asio_handler_deallocate(void* pointer, std::size_t size,
      binder2<Handler, Arg1, Arg2>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, &this_handler->handler_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
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

  friend void* asio_handler_allocate(std::size_t size,
      binder3<Handler, Arg1, Arg2, Arg3>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, &this_handler->handler_);
  }

  friend void asio_handler_deallocate(void* pointer, std::size_t size,
      binder3<Handler, Arg1, Arg2, Arg3>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, &this_handler->handler_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;
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

  friend void* asio_handler_allocate(std::size_t size,
      binder4<Handler, Arg1, Arg2, Arg3, Arg4>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, &this_handler->handler_);
  }

  friend void asio_handler_deallocate(void* pointer, std::size_t size,
      binder4<Handler, Arg1, Arg2, Arg3, Arg4>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, &this_handler->handler_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;
  Arg4 arg4_;
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

  friend void* asio_handler_allocate(std::size_t size,
      binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>* this_handler)
  {
    return asio_handler_alloc_helpers::allocate(
        size, &this_handler->handler_);
  }

  friend void asio_handler_deallocate(void* pointer, std::size_t size,
      binder5<Handler, Arg1, Arg2, Arg3, Arg4, Arg5>* this_handler)
  {
    asio_handler_alloc_helpers::deallocate(
        pointer, size, &this_handler->handler_);
  }

private:
  Handler handler_;
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;
  Arg4 arg4_;
  Arg5 arg5_;
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

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_BIND_HPP
