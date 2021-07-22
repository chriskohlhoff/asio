//
// async_result.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ARCHETYPES_ASYNC_RESULT_HPP
#define ARCHETYPES_ASYNC_RESULT_HPP

#include <asio/async_result.hpp>

namespace archetypes {

struct lazy_handler
{
};

template <typename Signature>
struct concrete_handler_1;

template <typename R, typename Arg1>
struct concrete_handler_1<R(Arg1)>
{
  concrete_handler_1(lazy_handler)
  {
  }

  void operator()(typename asio::decay<Arg1>::type)
  {
  }

#if defined(ASIO_HAS_MOVE)
  concrete_handler_1(concrete_handler_1&&) {}
private:
  concrete_handler_1(const concrete_handler_1&);
#endif // defined(ASIO_HAS_MOVE)
};

template <typename R, typename Arg1, typename Arg2>
struct concrete_handler_1<R(Arg1, Arg2)>
{
  concrete_handler_1(lazy_handler)
  {
  }

  void operator()(typename asio::decay<Arg1>::type, typename asio::decay<Arg2>::type)
  {
  }

#if defined(ASIO_HAS_MOVE)
  concrete_handler_1(concrete_handler_1&&) {}
private:
  concrete_handler_1(const concrete_handler_1&);
#endif // defined(ASIO_HAS_MOVE)
};

template <typename Signature1, typename Signature2>
struct concrete_handler_2;

template <typename R1, typename Arg1, typename R2, typename Arg2>
struct concrete_handler_2<R1(Arg1), R2(Arg2)>
{
  concrete_handler_2(lazy_handler)
  {
  }

  void operator()(typename asio::decay<Arg1>::type)
  {
  }

  void operator()(typename asio::decay<Arg2>::type)
  {
  }

#if defined(ASIO_HAS_MOVE)
  concrete_handler_2(concrete_handler_2&&) {}
private:
  concrete_handler_2(const concrete_handler_2&);
#endif // defined(ASIO_HAS_MOVE)
};

template <typename R1, typename Arg1, typename Arg2,
    typename R2, typename Arg3, typename Arg4>
struct concrete_handler_2<R1(Arg1, Arg2), R2(Arg3, Arg4)>
{
  concrete_handler_2(lazy_handler)
  {
  }

  void operator()(typename asio::decay<Arg1>::type, typename asio::decay<Arg2>::type)
  {
  }

  void operator()(typename asio::decay<Arg3>::type, typename asio::decay<Arg4>::type)
  {
  }

#if defined(ASIO_HAS_MOVE)
  concrete_handler_2(concrete_handler_2&&) {}
private:
  concrete_handler_2(const concrete_handler_2&);
#endif // defined(ASIO_HAS_MOVE)
};

} // namespace archetypes

namespace asio {

template <typename Signature>
class async_result<archetypes::lazy_handler, Signature>
{
public:
  // The concrete completion handler type.
  typedef archetypes::concrete_handler_1<Signature> completion_handler_type;

  // The return type of the initiating function.
  typedef int return_type;

  // Construct an async_result from a given handler.
  explicit async_result(completion_handler_type&)
  {
  }

  // Obtain the value to be returned from the initiating function.
  return_type get()
  {
    return 42;
  }

private:
  // Disallow copying and assignment.
  async_result(const async_result&) ASIO_DELETED;
  async_result& operator=(const async_result&) ASIO_DELETED;
};

template <typename Signature1, typename Signature2>
class async_result<archetypes::lazy_handler, Signature1, Signature2>
{
public:
  // The concrete completion handler type.
  typedef archetypes::concrete_handler_2<Signature1,
      Signature2> completion_handler_type;

  // The return type of the initiating function.
  typedef int return_type;

  // Construct an async_result from a given handler.
  explicit async_result(completion_handler_type&)
  {
  }

  // Obtain the value to be returned from the initiating function.
  return_type get()
  {
    return 42;
  }

private:
  // Disallow copying and assignment.
  async_result(const async_result&) ASIO_DELETED;
  async_result& operator=(const async_result&) ASIO_DELETED;
};

} // namespace asio

#endif // ARCHETYPES_ASYNC_RESULT_HPP
