//
// bind.hpp
// ~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_BIND_HPP
#define ASIO_DETAIL_BIND_HPP

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

// Some compilers (notably MSVC6) run into mysterious compiler errors when
// trying to use the boost::bind template in this library. The class and
// function templates below provide only the functionality of bind to create
// function objects with the signature void() as used in handlers passed to a
// demuxer's dispatch or post functions. This should make it simpler for the
// compiler to work correctly.

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
};

template <typename Handler, typename Arg1, typename Arg2, typename Arg3>
binder3<Handler, Arg1, Arg2, Arg3> bind_handler(Handler handler, Arg1 arg1,
    Arg2 arg2, Arg3 arg3)
{
  return binder3<Handler, Arg1, Arg2, Arg3>(handler, arg1, arg2, arg3);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_BIND_HPP
