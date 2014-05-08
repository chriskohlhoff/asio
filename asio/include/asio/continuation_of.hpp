//
// continuation_of.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_CONTINUATION_OF_HPP
#define ASIO_CONTINUATION_OF_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/result_type.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Argument>
struct continuation_signature
{
  typedef void signature(Argument);
};

template <>
struct continuation_signature<void>
{
  typedef void signature();
};

template <typename Result, typename Function, typename Continuation>
class continuation_chain
{
public:
  template <typename F, typename C>
  continuation_chain(ASIO_MOVE_ARG(F) f, ASIO_MOVE_ARG(C) c)
    : function_(ASIO_MOVE_CAST(F)(f)),
      continuation_(ASIO_MOVE_CAST(C)(c))
  {
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename... Args>
  void operator()(ASIO_MOVE_ARG(Args)... args)
  {
    continuation_(function_(ASIO_MOVE_CAST(Args)(args)...));
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void operator()()
  {
    continuation_(function_());
  }

#define ASIO_PRIVATE_CONTINUATION_CHAIN_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  void operator()(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    continuation_(function_(ASIO_VARIADIC_MOVE_ARGS(n))); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_CONTINUATION_CHAIN_DEF)
#undef ASIO_PRIVATE_CONTINUATION_CHAIN_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

private:
  Function function_;
  Continuation continuation_;
};

template <typename Function, typename Continuation>
class continuation_chain<void, Function, Continuation>
{
public:
  template <typename F, typename C>
  continuation_chain(ASIO_MOVE_ARG(F) f, ASIO_MOVE_ARG(C) c)
    : function_(ASIO_MOVE_CAST(F)(f)),
      continuation_(ASIO_MOVE_CAST(C)(c))
  {
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

  template <typename... Args>
  void operator()(ASIO_MOVE_ARG(Args)... args)
  {
    function_(ASIO_MOVE_CAST(Args)(args)...);
    continuation_();
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

  void operator()()
  {
    function_();
    continuation_();
  }

#define ASIO_PRIVATE_CONTINUATION_CHAIN_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  void operator()(ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    function_(ASIO_VARIADIC_MOVE_ARGS(n)); \
    continuation_(); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_CONTINUATION_CHAIN_DEF)
#undef ASIO_PRIVATE_CONTINUATION_CHAIN_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

private:
  Function function_;
  Continuation continuation_;
};

template <typename Function>
struct default_continuation_of_implementation
{
  typedef typename continuation_signature<
    typename result_type<Function>::type>::signature signature;

  template <typename F, typename C>
  static continuation_chain<typename result_type<Function>::type,
    Function, typename decay<C>::type> chain(
      ASIO_MOVE_ARG(F) f, ASIO_MOVE_ARG(C) c)
  {
    return continuation_chain<typename result_type<Function>::type,
      Function, typename decay<C>::type>(
        ASIO_MOVE_CAST(F)(f), ASIO_MOVE_CAST(C)(c));
  }
};

} // namespace detail

/// Type trait used to attach a continuation to a function.
/**
 * The <tt>continuation_of</tt> template enables customisation of function
 * invocation and passing of the result to a continuation.
 *
 * A program may specialise this template if the <tt>Function</tt> template
 * parameter in the specialisation is a user-defined type.
 *
 * Specialisations of continuation_of shall provide:
 *
 * @li A nested typedef <tt>signature</tt> that defines the required signature
 * for a function object that receives the result of a function of type
 * <tt>Function</tt>.
 *
 * @li A static member function <tt>chain</tt> that accepts two arguments
 * <tt>f</tt> and <tt>c</tt>, where <tt>f</tt> is an rvalue of type
 * <tt>Function</tt>, and <tt>c</tt> is a function object meeting
 * MoveConstructible requirements and matching the <tt>signature</tt>. The
 * <tt>chain</tt> member function returns a MoveConstructible function object
 * that accepts the same arguments as <tt>f</tt>.
 */
template <typename Function, typename = void>
struct continuation_of
#if !defined(GENERATING_DOCUMENTATION)
  : conditional<
      is_same<Function, typename decay<Function>::type>::value,
      asio::detail::default_continuation_of_implementation<Function>,
      continuation_of<typename decay<Function>::type> >::type
#endif // !defined(GENERATING_DOCUMENTATION)
{
#if defined(GENERATING_DOCUMENTATION)
  /// The signature of a continuation.
  /**
   * Type: <tt>void(R)</tt>, where <tt>R</tt> is determined as follows:
   *
   * @li if <tt>Function</tt> and <tt>decay<Function>::type</tt> are different
   * types, <tt>continuation_of<typename decay<Function>::type>::type</tt>;
   *
   * @li if <tt>Function</tt> is a function pointer type, the return type;
   *
   * @li if <tt>Function</tt> is a function object type with a single,
   * non-template overload of <tt>operator()</tt>, the return type of
   * <tt>operator()</tt> (N.B. requires C++11 <tt>decltype</tt>);
   *
   * @li if <tt>Function</tt> is a function object type with a nested type
   * <tt>result_type</tt>, <tt>result_type</tt>;
   *
   * @li otherwise, the program is ill-formed.
   */
  typedef see_below signature;

  /// Creates a new function object to attach a continuation.
  /**
   * If <tt>Function</tt> and <tt>decay<Function>::type</tt> are different
   * types, returns <tt>continuation_of<typename
   * decay<Function>::type>::chain(forward<F>(f), forward<C>(c))</tt>.
   * Otherwise, returns a function object that, when invoked, calls a copy of
   * <tt>f</tt>, and then passes the result to a copy of <tt>c</tt>.
   */
  template <typename F, typename C>
  static unspecified chain(F&& f, C&& c);
#endif // defined(GENERATING_DOCUMENTATION)
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_CONTINUATION_OF_HPP
