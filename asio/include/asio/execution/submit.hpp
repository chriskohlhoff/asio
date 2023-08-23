//
// execution/submit.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_SUBMIT_HPP
#define ASIO_EXECUTION_SUBMIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_DEPRECATED)

#include "asio/detail/type_traits.hpp"
#include "asio/execution/detail/submit_receiver.hpp"
#include "asio/execution/executor.hpp"
#include "asio/execution/receiver.hpp"
#include "asio/execution/sender.hpp"
#include "asio/execution/start.hpp"
#include "asio/traits/submit_member.hpp"
#include "asio/traits/submit_free.hpp"

#include "asio/detail/push_options.hpp"

#if defined(GENERATING_DOCUMENTATION)

namespace asio {
namespace execution {

/// A customisation point that submits a sender to a receiver.
/**
 * The name <tt>execution::submit</tt> denotes a customisation point object. For
 * some subexpressions <tt>s</tt> and <tt>r</tt>, let <tt>S</tt> be a type such
 * that <tt>decltype((s))</tt> is <tt>S</tt> and let <tt>R</tt> be a type such
 * that <tt>decltype((r))</tt> is <tt>R</tt>. The expression
 * <tt>execution::submit(s, r)</tt> is ill-formed if <tt>sender_to<S, R></tt> is
 * not <tt>true</tt>. Otherwise, it is expression-equivalent to:
 *
 * @li <tt>s.submit(r)</tt>, if that expression is valid and <tt>S</tt> models
 *   <tt>sender</tt>. If the function selected does not submit the receiver
 *   object <tt>r</tt> via the sender <tt>s</tt>, the program is ill-formed with
 *   no diagnostic required.
 *
 * @li Otherwise, <tt>submit(s, r)</tt>, if that expression is valid and
 *   <tt>S</tt> models <tt>sender</tt>, with overload resolution performed in a
 *   context that includes the declaration <tt>void submit();</tt> and that does
 *   not include a declaration of <tt>execution::submit</tt>. If the function
 *   selected by overload resolution does not submit the receiver object
 *   <tt>r</tt> via the sender <tt>s</tt>, the program is ill-formed with no
 *   diagnostic required.
 *
 * @li Otherwise, <tt>execution::start((new submit_receiver<S,
 *   R>{s,r})->state_)</tt>, where <tt>submit_receiver</tt> is an
 *   implementation-defined class template equivalent to:
 *   @code template<class S, class R>
 *   struct submit_receiver {
 *     struct wrap {
 *       submit_receiver * p_;
 *       template<class...As>
 *         requires receiver_of<R, As...>
 *       void set_value(As&&... as) &&
 *         noexcept(is_nothrow_receiver_of_v<R, As...>) {
 *         execution::set_value(std::move(p_->r_), (As&&) as...);
 *         delete p_;
 *       }
 *       template<class E>
 *         requires receiver<R, E>
 *       void set_error(E&& e) && noexcept {
 *         execution::set_error(std::move(p_->r_), (E&&) e);
 *         delete p_;
 *       }
 *       void set_done() && noexcept {
 *         execution::set_done(std::move(p_->r_));
 *         delete p_;
 *       }
 *     };
 *     remove_cvref_t<R> r_;
 *     connect_result_t<S, wrap> state_;
 *     submit_receiver(S&& s, R&& r)
 *       : r_((R&&) r)
 *       , state_(execution::connect((S&&) s, wrap{this})) {}
 *   };
 *   @endcode
 */
inline constexpr unspecified submit = unspecified;

/// A type trait that determines whether a @c submit expression is
/// well-formed.
/**
 * Class template @c can_submit is a trait that is derived from
 * @c true_type if the expression <tt>execution::submit(std::declval<R>(),
 * std::declval<E>())</tt> is well formed; otherwise @c false_type.
 */
template <typename S, typename R>
struct can_submit :
  integral_constant<bool, automatically_determined>
{
};

} // namespace execution
} // namespace asio

#else // defined(GENERATING_DOCUMENTATION)

namespace asio_execution_submit_fn {

using asio::declval;
using asio::enable_if_t;
using asio::execution::is_sender_to;
using asio::traits::submit_free;
using asio::traits::submit_member;

void submit();

enum overload_type
{
  call_member,
  call_free,
  adapter,
  ill_formed
};

template <typename S, typename R, typename = void,
    typename = void, typename = void>
struct call_traits
{
  static constexpr overload_type overload = ill_formed;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
};

template <typename S, typename R>
struct call_traits<S, void(R),
  enable_if_t<
    submit_member<S, R>::is_valid
  >,
  enable_if_t<
    is_sender_to<S, R>::value
  >> :
  submit_member<S, R>
{
  static constexpr overload_type overload = call_member;
};

template <typename S, typename R>
struct call_traits<S, void(R),
  enable_if_t<
    !submit_member<S, R>::is_valid
  >,
  enable_if_t<
    submit_free<S, R>::is_valid
  >,
  enable_if_t<
    is_sender_to<S, R>::value
  >> :
  submit_free<S, R>
{
  static constexpr overload_type overload = call_free;
};

template <typename S, typename R>
struct call_traits<S, void(R),
  enable_if_t<
    !submit_member<S, R>::is_valid
  >,
  enable_if_t<
    !submit_free<S, R>::is_valid
  >,
  enable_if_t<
    is_sender_to<S, R>::value
  >>
{
  static constexpr overload_type overload = adapter;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
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
    return static_cast<S&&>(s).submit(static_cast<R&&>(r));
  }

  template <typename S, typename R>
  constexpr enable_if_t<
    call_traits<S, void(R)>::overload == call_free,
    typename call_traits<S, void(R)>::result_type
  >
  operator()(S&& s, R&& r) const
    noexcept(call_traits<S, void(R)>::is_noexcept)
  {
    return submit(static_cast<S&&>(s), static_cast<R&&>(r));
  }

  template <typename S, typename R>
  constexpr enable_if_t<
    call_traits<S, void(R)>::overload == adapter,
    typename call_traits<S, void(R)>::result_type
  >
  operator()(S&& s, R&& r) const
    noexcept(call_traits<S, void(R)>::is_noexcept)
  {
    return asio::execution::start(
        (new asio::execution::detail::submit_receiver<S, R>(
          static_cast<S&&>(s), static_cast<R&&>(r)))->state_);
  }
};

template <typename T = impl>
struct static_instance
{
  static const T instance;
};

template <typename T>
const T static_instance<T>::instance = {};

} // namespace asio_execution_submit_fn
namespace asio {
namespace execution {
namespace {

static constexpr const asio_execution_submit_fn::impl&
  submit = asio_execution_submit_fn::static_instance<>::instance;

} // namespace

template <typename S, typename R>
struct can_submit :
  integral_constant<bool,
    asio_execution_submit_fn::call_traits<S, void(R)>::overload !=
      asio_execution_submit_fn::ill_formed>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
constexpr bool can_submit_v = can_submit<S, R>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
struct is_nothrow_submit :
  integral_constant<bool,
    asio_execution_submit_fn::call_traits<S, void(R)>::is_noexcept>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
constexpr bool is_nothrow_submit_v
  = is_nothrow_submit<S, R>::value;

#endif // defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
struct submit_result
{
  typedef typename asio_execution_submit_fn::call_traits<
      S, void(R)>::result_type type;
};

namespace detail {

template <typename S, typename R>
void submit_helper(S&& s, R&& r)
{
  execution::submit(static_cast<S&&>(s), static_cast<R&&>(r));
}

} // namespace detail
} // namespace execution
} // namespace asio

#endif // defined(GENERATING_DOCUMENTATION)

#include "asio/detail/pop_options.hpp"

#endif // !defined(ASIO_NO_DEPRECATED)

#endif // ASIO_EXECUTION_SUBMIT_HPP
