//
// execution/detail/bulk_sender.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2023 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_DETAIL_BULK_SENDER_HPP
#define ASIO_EXECUTION_DETAIL_BULK_SENDER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/connect.hpp"
#include "asio/execution/executor.hpp"
#include "asio/execution/set_done.hpp"
#include "asio/execution/set_error.hpp"
#include "asio/traits/connect_member.hpp"
#include "asio/traits/set_done_member.hpp"
#include "asio/traits/set_error_member.hpp"
#include "asio/traits/set_value_member.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename Receiver, typename Function, typename Number, typename Index>
struct bulk_receiver
{
  remove_cvref_t<Receiver> receiver_;
  decay_t<Function> f_;
  decay_t<Number> n_;

  template <typename R, typename F, typename N>
  explicit bulk_receiver(R&& r,
      F&& f, N&& n)
    : receiver_(static_cast<R&&>(r)),
      f_(static_cast<F&&>(f)),
      n_(static_cast<N&&>(n))
  {
  }

  void set_value()
  {
    for (Index i = 0; i < n_; ++i)
      f_(i);

    execution::set_value(static_cast<remove_cvref_t<Receiver>&&>(receiver_));
  }

  template <typename Error>
  void set_error(Error&& e) noexcept
  {
    execution::set_error(static_cast<remove_cvref_t<Receiver>&&>(receiver_),
        static_cast<Error&&>(e));
  }

  void set_done() noexcept
  {
    execution::set_done(static_cast<remove_cvref_t<Receiver>&&>(receiver_));
  }
};

template <typename Sender, typename Receiver,
  typename Function, typename Number>
struct bulk_receiver_traits
{
  typedef bulk_receiver<
      Receiver, Function, Number,
      typename execution::executor_index<
        remove_cvref_t<Sender>
      >::type
    > type;

  typedef type arg_type;
};

template <typename Sender, typename Function, typename Number>
struct bulk_sender : sender_base
{
  remove_cvref_t<Sender> sender_;
  decay_t<Function> f_;
  decay_t<Number> n_;

  template <typename S, typename F, typename N>
  explicit bulk_sender(S&& s,
      F&& f, N&& n)
    : sender_(static_cast<S&&>(s)),
      f_(static_cast<F&&>(f)),
      n_(static_cast<N&&>(n))
  {
  }

  template <typename Receiver>
  connect_result_t<
      remove_cvref_t<Sender>,
      typename bulk_receiver_traits<
        Sender, Receiver, Function, Number
      >::arg_type
  > connect(Receiver&& r,
      enable_if_t<
        can_connect<
          remove_cvref_t<Sender>&&,
          typename bulk_receiver_traits<
            Sender, Receiver, Function, Number
          >::arg_type
        >::value
      >* = 0) && noexcept
  {
    return execution::connect(
        static_cast<remove_cvref_t<Sender>&&>(sender_),
        typename bulk_receiver_traits<Sender, Receiver, Function, Number>::type(
          static_cast<Receiver&&>(r),
          static_cast<decay_t<Function>&&>(f_),
          static_cast<decay_t<Number>&&>(n_)));
  }

  template <typename Receiver>
  connect_result_t<
      const remove_cvref_t<Sender>&,
      typename bulk_receiver_traits<
        Sender, Receiver, Function, Number
      >::arg_type
  > connect(Receiver&& r,
      enable_if_t<
        can_connect<
          const remove_cvref_t<Sender>&,
          typename bulk_receiver_traits<
            Sender, Receiver, Function, Number
          >::arg_type
        >::value
      >* = 0) const & noexcept
  {
    return execution::connect(sender_,
        typename bulk_receiver_traits<Sender, Receiver, Function, Number>::type(
          static_cast<Receiver&&>(r), f_, n_));
  }
};

} // namespace detail
} // namespace execution
namespace traits {

#if !defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

template <typename Receiver, typename Function, typename Number, typename Index>
struct set_value_member<
    execution::detail::bulk_receiver<Receiver, Function, Number, Index>,
    void()>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)

template <typename Receiver, typename Function,
    typename Number, typename Index, typename Error>
struct set_error_member<
    execution::detail::bulk_receiver<Receiver, Function, Number, Index>,
    Error>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_SET_DONE_MEMBER_TRAIT)

template <typename Receiver, typename Function, typename Number, typename Index>
struct set_done_member<
    execution::detail::bulk_receiver<Receiver, Function, Number, Index>>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = true;
  typedef void result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_SET_DONE_MEMBER_TRAIT)

#if !defined(ASIO_HAS_DEDUCED_CONNECT_MEMBER_TRAIT)

template <typename Sender, typename Function,
    typename Number, typename Receiver>
struct connect_member<
    execution::detail::bulk_sender<Sender, Function, Number>,
    Receiver,
    enable_if_t<
      execution::can_connect<
        remove_cvref_t<Sender>,
        typename execution::detail::bulk_receiver_traits<
          Sender, Receiver, Function, Number
        >::arg_type
      >::value
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef execution::connect_result_t<
      remove_cvref_t<Sender>,
      typename execution::detail::bulk_receiver_traits<
        Sender, Receiver, Function, Number
      >::arg_type
    > result_type;
};

template <typename Sender, typename Function,
    typename Number, typename Receiver>
struct connect_member<
    const execution::detail::bulk_sender<Sender, Function, Number>,
    Receiver,
    enable_if_t<
      execution::can_connect<
        const remove_cvref_t<Sender>&,
        typename execution::detail::bulk_receiver_traits<
          Sender, Receiver, Function, Number
        >::arg_type
      >::value
    >>
{
  static constexpr bool is_valid = true;
  static constexpr bool is_noexcept = false;
  typedef execution::connect_result_t<
      const remove_cvref_t<Sender>&,
      typename execution::detail::bulk_receiver_traits<
        Sender, Receiver, Function, Number
      >::arg_type
    > result_type;
};

#endif // !defined(ASIO_HAS_DEDUCED_CONNECT_MEMBER_TRAIT)

} // namespace traits
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_DETAIL_BULK_SENDER_HPP
