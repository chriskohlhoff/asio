//
// execution/detail/enumeration.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_DETAIL_ENUMERATION_HPP
#define ASIO_EXECUTION_DETAIL_ENUMERATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/can_query.hpp"
#include "asio/execution/detail/query_member_traits.hpp"
#include "asio/execution/detail/query_static_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {
namespace detail {

template <typename Derived, int N, bool Defaulted>
struct enumeration
{
  template <int Value>
  struct enumerator;

  template <typename Enumerator>
  struct default_enumerator
  {
    template <typename Executor>
    static constexpr auto static_query()
      -> decltype(Executor::query(*static_cast<Enumerator*>(0)))
    {
      return Executor::query(Enumerator());
    }

    template <typename Executor, int I = 1>
    struct use_default_static_query :
      conditional<
        can_query<Executor, enumerator<I> >::value,
        false_type,
        typename conditional<
          I + 1 < N,
          use_default_static_query<Executor, I + 1>,
          true_type
        >::type
      >::type
    {
    };

    template <typename Executor>
    static constexpr auto static_query()
      -> typename enable_if<
        !query_member_traits<Executor, Enumerator>::is_valid
          && use_default_static_query<Executor>::value,
        Enumerator
      >::type
    {
      return Enumerator();
    }
  };

  template <typename Enumerator>
  struct non_default_enumerator
  {
    template <typename Executor>
    static constexpr auto static_query()
      -> decltype(Executor::query(*static_cast<Enumerator*>(0)))
    {
      return Executor::query(Enumerator());
    }
  };

  template <int Value>
  struct enumerator :
    conditional<
      Defaulted && Value == 0,
      default_enumerator<enumerator<Value> >,
      non_default_enumerator<enumerator<Value> >
    >::type
  {
    static constexpr bool is_requirable = true;
    static constexpr bool is_preferable = true;

    using polymorphic_query_result_type = Derived;

    template <typename Executor,
      typename T = decltype(enumerator::template static_query<Executor>())>
        static constexpr T static_query_v
          = enumerator::template static_query<Executor>();

    static constexpr Derived value()
    {
      return Derived(Value);
    }
  };

  static constexpr bool is_requirable = false;
  static constexpr bool is_preferable = false;

  using polymorphic_query_result_type = Derived;

  template <typename Executor>
  static constexpr auto static_query() ->
    decltype(Executor::query(*static_cast<Derived*>(0)))
  {
    return Executor::query(Derived());
  }

  template <typename Executor, int I = 0>
  struct static_query_type :
    conditional<
      query_static_traits<Executor, enumerator<I> >::is_valid,
      decay<enumerator<I> >,
      typename conditional<
        I + 1 < N,
        static_query_type<Executor, I + 1>,
        decay<enable_if<false> >
      >::type
    >::type
  {
  };

  template <typename Executor>
  static constexpr auto static_query() ->
    typename enable_if<
      !query_member_traits<Executor, Derived>::is_valid,
      decltype(
        static_query_type<Executor>::type::template static_query_v<Executor>)
    >::type
  {
    return static_query_type<Executor>::type::template static_query_v<Executor>;
  }

  template <typename Executor,
    typename T = decltype(enumeration::static_query<Executor>())>
      static constexpr T static_query_v
        = enumeration::static_query<Executor>();

  constexpr enumeration()
    : value_(-1)
  {
  }

  template <int I, typename = typename enable_if<I < N>::type>
  constexpr enumeration(enumerator<I>)
    : value_(enumerator<I>::value().value_)
  {
  }

  template <typename Executor, int I = 0>
  struct query_type :
    conditional<
      can_query<Executor, enumerator<I> >::value,
      decay<enumerator<I> >,
      typename conditional<
        I + 1 < N,
        query_type<Executor, I + 1>,
        decay<enable_if<false> >
      >::type
    >::type
  {
  };

  template <typename Executor, typename Property,
    typename = typename enable_if<is_same<Property, Derived>::value>::type>
  friend constexpr auto query(const Executor& ex, const Property& p)
    noexcept(noexcept(execution::query(ex,
      typename query_type<Executor>::type())))
    -> decltype(execution::query(ex, typename query_type<Executor>::type()))
  {
    return execution::query(ex, typename query_type<Executor>::type());
  }

  friend constexpr bool operator==(const Derived& a, const Derived& b) noexcept
  {
    return a.value_ == b.value_;
  }

  friend constexpr bool operator!=(const Derived& a, const Derived& b) noexcept
  {
    return a.value_ != b.value_;
  }

private:
  constexpr enumeration(int v)
    : value_(v)
  {
  }

  int value_;
};

} // namespace detail
} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_DETAIL_ENUMERATION_HPP
