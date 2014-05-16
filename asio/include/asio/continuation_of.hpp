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

template <typename Function, typename ResultOfArgs>
struct default_continuation_of
{
  typedef typename continuation_signature<
    typename result_of<ResultOfArgs>::type>::signature signature;

  template <typename C>
  struct chain_type
  {
    typedef continuation_chain<typename result_of<ResultOfArgs>::type,
      Function, typename decay<C>::type> type;
  };

  template <typename F, typename C>
  static typename chain_type<C>::type chain(
      ASIO_MOVE_ARG(F) f, ASIO_MOVE_ARG(C) c)
  {
    return typename chain_type<C>::type(
        ASIO_MOVE_CAST(F)(f), ASIO_MOVE_CAST(C)(c));
  }
};

} // namespace detail

/// Not defined.
template <typename>
struct continuation_of;

#if defined(GENERATING_DOCUMENTATION)

/// Type trait used to attach a continuation to a function.
/**
 * The <tt>continuation_of</tt> template enables customisation of function
 * invocation and passing of the result to a continuation.
 *
 * A program may specialise this template for a user-defined <tt>Function</tt>
 * type.
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
template <typename Function, typename... Args>
struct continuation_of<Function(Args...)>
{
  /// The signature of a continuation.
  /**
   * Type:
   *
   * @li If <tt>Function</tt> and <tt>decay<Function>::type</tt> are different
   * types, <tt>continuation_of<typename decay<Function>::type>::type</tt>;
   *
   * @li Let <tt>R</tt> be the type produced by <tt>typename
   * result_of<Function(Args...)>::type</tt>. If <tt>R</tt> is <tt>void</tt>,
   * <tt>void()</tt>. If <tt>R</tt> is non-void, <tt>void(R)</tt>.
   *
   * @li Otherwise, if <tt>result_of<Function(Args...)></tt> does not contain a
   * nested type named <tt>type/tt>, the program is ill-formed.
   */
  typedef see_below signature;

  /// Defines the type produced by the @c chain() function for a type @c C.
  template <typename C>
  struct chain_type
  {
    typedef unspecified type;
  };

  /// Creates a new function object to attach a continuation.
  /**
   * If <tt>Function</tt> and <tt>decay<Function>::type</tt> are different
   * types, returns <tt>continuation_of<typename
   * decay<Function>::type>::chain(forward<F>(f), forward<C>(c))</tt>.
   * Otherwise, returns a function object that, when invoked, calls a copy of
   * <tt>f</tt>, and then passes the result to a copy of <tt>c</tt>.
   */
  template <typename F, typename C>
  static typename chain_type<C>::type chain(F&& f, C&& c);
};

#elif defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Function, typename... Args>
struct continuation_of<Function(Args...)>
  : conditional<
      is_same<Function, typename decay<Function>::type>::value,
      asio::detail::default_continuation_of<Function, Function(Args...)>,
      continuation_of<typename decay<Function>::type> >::type {};

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Function>
struct continuation_of<Function()>
  : conditional<
      is_same<Function, typename decay<Function>::type>::value,
      asio::detail::default_continuation_of<Function, Function()>,
      continuation_of<typename decay<Function>::type> >::type {};

# define ASIO_PRIVATE_CONTINUATION_OF_DEF(n) \
  template <typename Function, ASIO_VARIADIC_TPARAMS(n)> \
  struct continuation_of<Function(ASIO_VARIADIC_TARGS(n))> \
    : conditional< \
        is_same<Function, typename decay<Function>::type>::value, \
        asio::detail::default_continuation_of< \
          Function, Function(ASIO_VARIADIC_TARGS(n))>, \
        continuation_of<typename decay<Function>::type> >::type {}; \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_CONTINUATION_OF_DEF)
#undef ASIO_PRIVATE_CONTINUATION_OF_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_CONTINUATION_OF_HPP
