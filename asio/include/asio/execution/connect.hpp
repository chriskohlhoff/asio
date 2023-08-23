//
// execution/connect.hpp
// ~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_CONNECT_HPP
#define ASIO_EXECUTION_CONNECT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_DEPRECATED)

#include "asio/detail/type_traits.hpp"
#include "asio/execution/detail/as_invocable.hpp"
#include "asio/execution/detail/as_operation.hpp"
#include "asio/execution/detail/as_receiver.hpp"
#include "asio/execution/executor.hpp"
#include "asio/execution/operation_state.hpp"
#include "asio/execution/receiver.hpp"
#include "asio/execution/sender.hpp"
#include "asio/traits/connect_member.hpp"
#include "asio/traits/connect_free.hpp"

#include "asio/detail/push_options.hpp"

#if defined(GENERATING_DOCUMENTATION)

namespace asio {
namespace execution {

/// A customisation point that connects a sender to a receiver.
/**
 * The name <tt>execution::connect</tt> denotes a customisation point object.
 * For some subexpressions <tt>s</tt> and <tt>r</tt>, let <tt>S</tt> be a type
 * such that <tt>decltype((s))</tt> is <tt>S</tt> and let <tt>R</tt> be a type
 * such that <tt>decltype((r))</tt> is <tt>R</tt>. The expression
 * <tt>execution::connect(s, r)</tt> is expression-equivalent to:
 *
 * @li <tt>s.connect(r)</tt>, if that expression is valid, if its type
 *   satisfies <tt>operation_state</tt>, and if <tt>S</tt> satisfies
 *   <tt>sender</tt>.
 *
 * @li Otherwise, <tt>connect(s, r)</tt>, if that expression is valid, if its
 *   type satisfies <tt>operation_state</tt>, and if <tt>S</tt> satisfies
 *   <tt>sender</tt>, with overload resolution performed in a context that
 *   includes the declaration <tt>void connect();</tt> and that does not include
 *   a declaration of <tt>execution::connect</tt>.
 *
 * @li Otherwise, <tt>as_operation{s, r}</tt>, if <tt>r</tt> is not an instance
 *  of <tt>as_receiver<F, S></tt> for some type <tt>F</tt>, and if
 *  <tt>receiver_of<R> && executor_of<remove_cvref_t<S>,
 *  as_invocable<remove_cvref_t<R>, S>></tt> is <tt>true</tt>, where
 *  <tt>as_operation</tt> is an implementation-defined class equivalent to
 *  @code template <class S, class R>
 *  struct as_operation
 *  {
 *    remove_cvref_t<S> e_;
 *    remove_cvref_t<R> r_;
 *    void start() noexcept try {
 *      execution::execute(std::move(e_),
 *          as_invocable<remove_cvref_t<R>, S>{r_});
 *    } catch(...) {
 *      execution::set_error(std::move(r_), current_exception());
 *    }
 *  }; @endcode
 *  and <tt>as_invocable</tt> is a class template equivalent to the following:
 *  @code template<class R>
 *  struct as_invocable
 *  {
 *    R* r_;
 *    explicit as_invocable(R& r) noexcept
 *      : r_(std::addressof(r)) {}
 *    as_invocable(as_invocable && other) noexcept
 *      : r_(std::exchange(other.r_, nullptr)) {}
 *    ~as_invocable() {
 *      if(r_)
 *        execution::set_done(std::move(*r_));
 *    }
 *    void operator()() & noexcept try {
 *      execution::set_value(std::move(*r_));
 *      r_ = nullptr;
 *    } catch(...) {
 *      execution::set_error(std::move(*r_), current_exception());
 *      r_ = nullptr;
 *    }
 *  };
 *  @endcode
 *
 * @li Otherwise, <tt>execution::connect(s, r)</tt> is ill-formed.
 */
inline constexpr unspecified connect = unspecified;

/// A type trait that determines whether a @c connect expression is
/// well-formed.
/**
 * Class template @c can_connect is a trait that is derived from
 * @c true_type if the expression <tt>execution::connect(std::declval<S>(),
 * std::declval<R>())</tt> is well formed; otherwise @c false_type.
 */
template <typename S, typename R>
struct can_connect :
  integral_constant<bool, automatically_determined>
{
};

/// A type trait to determine the result of a @c connect expression.
template <typename S, typename R>
struct connect_result
{
  /// The type of the connect expression.
  /**
   * The type of the expression <tt>execution::connect(std::declval<S>(),
   * std::declval<R>())</tt>.
   */
  typedef automatically_determined type;
};

/// A type alis to determine the result of a @c connect expression.
template <typename S, typename R>
using connect_result_t = typename connect_result<S, R>::type;

} // namespace execution
} // namespace asio

#else // defined(GENERATING_DOCUMENTATION)

namespace asio_execution_connect_fn {

using asio::conditional_t;
using asio::declval;
using asio::enable_if_t;
using asio::execution::detail::as_invocable;
using asio::execution::detail::as_operation;
using asio::execution::detail::is_as_receiver;
using asio::execution::is_executor_of;
using asio::execution::is_operation_state;
using asio::execution::is_receiver;
using asio::execution::is_sender;
using asio::false_type;
using asio::remove_cvref_t;
using asio::traits::connect_free;
using asio::traits::connect_member;

void connect();

enum overload_type
{
  call_member,
  call_free,
  adapter,
  ill_formed
};

template <typename S, typename R, typename = void,
   typename = void, typename = void, typename = void>
struct call_traits
{
  static constexpr overload_type overload = ill_formed;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
};

template <typename S, typename R>
struct call_traits<S, void(R),
  enable_if_t<
    connect_member<S, R>::is_valid
  >,
  enable_if_t<
    is_operation_state<typename connect_member<S, R>::result_type>::value
  >,
  enable_if_t<
    is_sender<remove_cvref_t<S>>::value
  >> :
  connect_member<S, R>
{
  static constexpr overload_type overload = call_member;
};

template <typename S, typename R>
struct call_traits<S, void(R),
  enable_if_t<
    !connect_member<S, R>::is_valid
  >,
  enable_if_t<
    connect_free<S, R>::is_valid
  >,
  enable_if_t<
    is_operation_state<typename connect_free<S, R>::result_type>::value
  >,
  enable_if_t<
    is_sender<remove_cvref_t<S>>::value
  >> :
  connect_free<S, R>
{
  static constexpr overload_type overload = call_free;
};

template <typename S, typename R>
struct call_traits<S, void(R),
  enable_if_t<
    !connect_member<S, R>::is_valid
  >,
  enable_if_t<
    !connect_free<S, R>::is_valid
  >,
  enable_if_t<
    is_receiver<R>::value
  >,
  enable_if_t<
    conditional_t<
      !is_as_receiver<
        remove_cvref_t<R>
      >::value,
      is_executor_of<
        remove_cvref_t<S>,
        as_invocable<remove_cvref_t<R>, S>
      >,
      false_type
    >::value
  >>
{
  static constexpr overload_type overload = adapter;
  static constexpr bool is_noexcept = false;
  typedef as_operation<S, R> result_type;
};

struct impl
{
  template <typename S, typename R>
  constexpr enable_if_t<
    call_traits<S, void(R)>::overload == call_member,
    typename call_traits<S, void(R)>::result_type
  >
  operator()(S&& s, R&& r) const
    noexcept(call_traits<S, void(R)>::is_noexcept)
  {
    return static_cast<S&&>(s).connect(static_cast<R&&>(r));
  }

  template <typename S, typename R>
  constexpr enable_if_t<
    call_traits<S, void(R)>::overload == call_free,
    typename call_traits<S, void(R)>::result_type
  >
  operator()(S&& s, R&& r) const
    noexcept(call_traits<S, void(R)>::is_noexcept)
  {
    return connect(static_cast<S&&>(s), static_cast<R&&>(r));
  }

  template <typename S, typename R>
  constexpr enable_if_t<
    call_traits<S, void(R)>::overload == adapter,
    typename call_traits<S, void(R)>::result_type
  >
  operator()(S&& s, R&& r) const
    noexcept(call_traits<S, void(R)>::is_noexcept)
  {
    return typename call_traits<S, void(R)>::result_type(
        static_cast<S&&>(s), static_cast<R&&>(r));
  }
};

template <typename T = impl>
struct static_instance
{
  static const T instance;
};

template <typename T>
const T static_instance<T>::instance = {};

} // namespace asio_execution_connect_fn
namespace asio {
namespace execution {
namespace {

static constexpr const asio_execution_connect_fn::impl&
  connect = asio_execution_connect_fn::static_instance<>::instance;

} // namespace

template <typename S, typename R>
struct can_connect :
  integral_constant<bool,
    asio_execution_connect_fn::call_traits<S, void(R)>::overload !=
      asio_execution_connect_fn::ill_formed>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
constexpr bool can_connect_v = can_connect<S, R>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
struct is_nothrow_connect :
  integral_constant<bool,
    asio_execution_connect_fn::call_traits<S, void(R)>::is_noexcept>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
constexpr bool is_nothrow_connect_v = is_nothrow_connect<S, R>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
struct connect_result
{
  typedef typename asio_execution_connect_fn::call_traits<
      S, void(R)>::result_type type;
};

template <typename S, typename R>
using connect_result_t = typename connect_result<S, R>::type;

} // namespace execution
} // namespace asio

#endif // defined(GENERATING_DOCUMENTATION)

#include "asio/detail/pop_options.hpp"

#endif // !defined(ASIO_NO_DEPRECATED)

#endif // ASIO_EXECUTION_CONNECT_HPP
