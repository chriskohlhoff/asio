//
// executor_wrapper.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2014 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTOR_WRAPPER_HPP
#define ASIO_EXECUTOR_WRAPPER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/result_type.hpp"
#include "asio/detail/variadic_templates.hpp"
#include "asio/async_result.hpp"
#include "asio/continuation_of.hpp"
#include "asio/handler_type.hpp"
#include "asio/uses_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
struct executor_wrapper_check
{
  typedef void type;
};

template <typename T, typename = void>
struct executor_wrapper_result_type {};

template <typename T>
struct executor_wrapper_result_type<T,
  typename executor_wrapper_check<typename result_type<T>::type>::type>
{
  typedef typename result_type<T>::type result_type;
};

template <typename T, typename Executor, bool UsesExecutor>
class executor_wrapper_base;

template <typename T, typename Executor>
class executor_wrapper_base<T, Executor, true>
{
protected:
  template <typename E, typename U>
  executor_wrapper_base(ASIO_MOVE_ARG(E) e, ASIO_MOVE_ARG(U) u)
    : wrapped_(executor_arg_t(),
        ASIO_MOVE_CAST(E)(e), ASIO_MOVE_CAST(U)(u))
  {
  }

  Executor get_executor_base() const
  {
    return wrapped_.get_executor();
  }

  T wrapped_;
};

template <typename T, typename Executor>
class executor_wrapper_base<T, Executor, false>
  : private Executor
{
protected:
  template <typename E, typename U>
  executor_wrapper_base(ASIO_MOVE_ARG(E) e, ASIO_MOVE_ARG(U) u)
    : Executor(ASIO_MOVE_CAST(E)(e)),
      wrapped_(ASIO_MOVE_CAST(U)(u))
  {
  }

  Executor get_executor_base() const
  {
    return static_cast<const Executor&>(*this);
  }

  T wrapped_;
};

} // namespace detail

/// A call wrapper type to associate an object of type @c T with an executor of
/// type @c Executor.
template <typename T, typename Executor>
class executor_wrapper
#if !defined(GENERATING_DOCUMENTATION)
  : public detail::executor_wrapper_result_type<T>,
    private detail::executor_wrapper_base<
      T, Executor, uses_executor<T, Executor>::value>
#endif // !defined(GENERATING_DOCUMENTATION)
{
public:
  /// The type of the associated executor.
  typedef Executor executor_type;

#if defined(GENERATING_DOCUMENTATION)
  /// The return type if a function.
  /**
   * The type of @c result_type is based on the type @c T of the wrapper's
   * target object:
   *
   * @li if @c T is a pointer to function type, @c result_type is a synonym for
   * the return type of @c T;
   *
   * @li if @c T is a pointer to member function, @c result_type is a synonym
   * for the return type of @c T;
   *
   * @li if @c T is a class type and has a single, non-template overload of
   * <tt>operator()</tt>, @c result_type is a synonym for the return type of
   * <tt>T::operator()</tt>.
   *
   * @li if @c T is a class type with a member type @c result_type, then @c
   * result_type is a synonym for @c T::result_type;
   *
   * @li otherwise @c result_type is not defined.
   */
  typedef see_below result_type;
#endif // defined(GENERATING_DOCUMENTATION)

  /// Construct an executor wrapper for the specified object.
  /**
   * This constructor is only valid if the @c Executor type is default
   * constructible, and the type @c T is constructible from type @c U.
   */
  template <typename U>
  explicit executor_wrapper(ASIO_MOVE_ARG(U) u)
    : base_type(Executor(), ASIO_MOVE_CAST(U)(u))
  {
  }

  /// Construct an executor wrapper for the specified object.
  /**
   * This constructor is only valid if the type @c T is constructible from type
   * @c U.
   */
  template <typename U>
  executor_wrapper(executor_arg_t, const executor_type& e,
      ASIO_MOVE_ARG(U) u)
    : base_type(e, ASIO_MOVE_CAST(U)(u))
  {
  }

  /// Copy constructor.
  executor_wrapper(const executor_wrapper& other)
    : base_type(other.executor_, other.wrapped_)
  {
  }

  /// Construct a copy, but specify a different executor.
  executor_wrapper(executor_arg_t, const executor_type& e,
      const executor_wrapper& other)
    : base_type(e, other.wrapped_)
  {
  }

  /// Construct a copy of a different executor wrapper type.
  /**
   * This constructor is only valid if the @c Executor type is constructible
   * from type @c OtherExecutor, and the type @c T is constructible from type
   * @c U.
   */
  template <typename U, typename OtherExecutor>
  executor_wrapper(const executor_wrapper<U, OtherExecutor>& other)
    : base_type(other.executor_, other.wrapped_)
  {
  }

  /// Construct a copy of a different executor wrapper type, but specify a
  /// different executor.
  /**
   * This constructor is only valid if the type @c T is constructible from type
   * @c U.
   */
  template <typename U, typename OtherExecutor>
  executor_wrapper(executor_arg_t, const executor_type& e,
      const executor_wrapper<U, OtherExecutor>& other)
    : base_type(e, other.wrapped_)
  {
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Move constructor.
  executor_wrapper(executor_wrapper&& other)
    : base_type(ASIO_MOVE_CAST(executor_type)(other.get_executor()),
        ASIO_MOVE_CAST(T)(other.wrapped_))
  {
  }

  /// Move construct the wrapped object, but specify a different executor.
  executor_wrapper(executor_arg_t, const executor_type& e,
      executor_wrapper&& other)
    : base_type(e, ASIO_MOVE_CAST(T)(other.wrapped_))
  {
  }

  /// Move construct from a different executor wrapper type.
  template <typename U, typename OtherExecutor>
  executor_wrapper(executor_wrapper<U, OtherExecutor>&& other)
    : base_type(ASIO_MOVE_CAST(OtherExecutor)(other.executor_),
        ASIO_MOVE_CAST(U)(other.wrapped_))
  {
  }

  /// Move construct from a different executor wrapper type, but specify a
  /// different executor.
  template <typename U, typename OtherExecutor>
  executor_wrapper(executor_arg_t, const executor_type& e,
      executor_wrapper<U, OtherExecutor>&& other)
    : base_type(e, ASIO_MOVE_CAST(U)(other.wrapped_))
  {
  }

#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Destructor.
  ~executor_wrapper()
  {
  }

  /// Obtain the associated executor.
  executor_type get_executor() const
  {
    return this->get_executor_base();
  }

#if defined(ASIO_HAS_VARIADIC_TEMPLATES) \
  || defined(GENERATING_DOCUMENTATION)

  /// Forwarding function call operator.
  template <typename... Args>
  typename result_of<T(Args...)>::type operator()(
      ASIO_MOVE_ARG(Args)... args)
  {
    return this->wrapped_(ASIO_MOVE_CAST(Args)(args)...);
  }

  /// Forwarding function call operator.
  template <typename... Args>
  typename result_of<T(Args...)>::type operator()(
      ASIO_MOVE_ARG(Args)... args) const
  {
    return this->wrapped_(ASIO_MOVE_CAST(Args)(args)...);
  }

#else // defined(ASIO_HAS_VARIADIC_TEMPLATES)
      //   || defined(GENERATING_DOCUMENTATION)

  typename result_of<T()>::type operator()()
  {
    return this->wrapped_();
  }

  typename result_of<T()>::type operator()() const
  {
    return this->wrapped_();
  }

#define ASIO_PRIVATE_EXECUTOR_WRAPPER_CALL_DEF(n) \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  typename result_of<T(ASIO_VARIADIC_TARGS(n))>::type operator()( \
      ASIO_VARIADIC_MOVE_PARAMS(n)) \
  { \
    return this->wrapped_(ASIO_VARIADIC_MOVE_ARGS(n)); \
  } \
  \
  template <ASIO_VARIADIC_TPARAMS(n)> \
  typename result_of<T(ASIO_VARIADIC_TARGS(n))>::type operator()( \
      ASIO_VARIADIC_MOVE_PARAMS(n)) const \
  { \
    return this->wrapped_(ASIO_VARIADIC_MOVE_ARGS(n)); \
  } \
  /**/
  ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_EXECUTOR_WRAPPER_CALL_DEF)
#undef ASIO_PRIVATE_EXECUTOR_WRAPPER_CALL_DEF

#endif // defined(ASIO_HAS_VARIADIC_TEMPLATES)
       //   || defined(GENERATING_DOCUMENTATION)

private:
  typedef detail::executor_wrapper_base<T, Executor,
    uses_executor<T, Executor>::value> base_type;
  template <typename, typename> friend class executor_wrapper;
  friend class async_result<executor_wrapper>;
  friend class continuation_of<executor_wrapper>;
};

/// Determine the type to use when associating an executor of type @c Executor
/// with an object of type @c T.
template <typename T, typename Executor>
struct wrap_with_executor_type
{
  /// The wrapper type.
  /**
   * Let @c DecayT be the type <tt>typename decay<T>::type</tt>. If
   * <tt>uses_executor<DecayT, Executor>::value</tt> is false, @c type is a
   * synonym for <tt>executor_wrapper<DecayT, Executor></tt>. Otherwise, @c
   * type is a synonym for @c DecayT.
   */
  typedef typename conditional<
    uses_executor<typename decay<T>::type, Executor>::value,
    typename decay<T>::type,
    executor_wrapper<typename decay<T>::type, Executor> >::type type;
};

/// Associate an object of type @c T with an executor of type @c Executor.
/**
 * Let @c DecayT be the type <tt>typename decay<T>::type</tt>. If
 * <tt>uses_executor<DecayT, Executor>::value</tt> is false, returns an object
 * of type <tt>executor_wrapper<DecayT, Executor></tt>. Otherwise, returns a
 * copy of @c t constructed as <tt>DecayT(executor_arg, e, t)</tt>.
 */
template <typename T, typename Executor>
inline typename wrap_with_executor_type<T, Executor>::type
wrap_with_executor(ASIO_MOVE_ARG(T) t, const Executor& e)
{
  return typename wrap_with_executor_type<T, Executor>::type(
      executor_arg_t(), e, ASIO_MOVE_CAST(T)(t));
}

#if !defined(GENERATING_DOCUMENTATION)

template <typename T, typename Executor>
struct uses_executor<executor_wrapper<T, Executor>, Executor>
  : true_type {};

template <typename T, typename Executor, typename Signature>
struct handler_type<executor_wrapper<T, Executor>, Signature>
{
  typedef executor_wrapper<
    typename handler_type<T, Signature>::type, Executor> type;
};

template <typename T, typename Executor>
class async_result<executor_wrapper<T, Executor> >
{
public:
  typedef typename async_result<T>::type type;

  explicit async_result(executor_wrapper<T, Executor>& w)
    : wrapped_(w.wrapped_)
  {
  }

  type get()
  {
    return wrapped_.get();
  }

private:
  async_result<T> wrapped_;
};

template <typename T, typename Executor>
struct continuation_of<executor_wrapper<T, Executor> >
{
  typedef typename continuation_of<T>::signature signature;

  template <typename C>
  struct chain_type
  {
    typedef executor_wrapper<
      typename continuation_of<T>::template chain_type<C>::type,
      Executor> type;
  };

  template <typename C>
  static typename chain_type<C>::type chain(
      executor_wrapper<T, Executor> w, ASIO_MOVE_ARG(C) c)
  {
    return typename chain_type<C>::type(executor_arg_t(), w.get_executor(),
        continuation_of<T>::chain(w.wrapped_, ASIO_MOVE_CAST(C)(c)));
  }
};

#endif // !defined(GENERATING_DOCUMENTATION)

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTOR_WRAPPER_HPP
