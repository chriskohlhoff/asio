//
// execution/blocking_adaptation.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_BLOCKING_ADAPTATION_HPP
#define ASIO_EXECUTION_BLOCKING_ADAPTATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/execution/detail/enumeration.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct blocking_adaptation_t :
  detail::enumeration<blocking_adaptation_t, 2>
{
  using detail::enumeration<blocking_adaptation_t, 2>::enumeration;

  using disallowed_t = enumerator<0>;
  using allowed_t = enumerator<1>;

  static constexpr disallowed_t disallowed{};
  static constexpr allowed_t allowed{};

  template <typename Executor>
  struct adapter
  {
  public:
    adapter(Executor ex)
      : executor_(std::move(ex))
    {
    }

    template <typename Property>
    auto require(const Property& p) const
      -> adapter<typename decay<
        decltype(declval<typename conditional<true,
          Executor, Property>::type>().require(p))>::type>
    {
      return adapter<typename decay<
        decltype(declval<typename conditional<true,
          Executor, Property>::type>().require(p))>::type>(
            executor_.require(p));
    }

    static constexpr blocking_adaptation_t query(
        const blocking_adaptation_t&) noexcept
    {
      return allowed_t{};
    }

    template <typename Property>
    auto query(const Property& p) const
      noexcept(noexcept(declval<typename conditional<true,
            Executor, Property>::type>().query(p)))
      -> decltype(declval<typename conditional<true,
          Executor, Property>::type>().query(p))
    {
      return executor_.query(p);
    }

    template <typename Function>
    auto execute(Function&& f) const
      -> decltype(declval<typename conditional<true,
          Executor, Function>::type>().execute(declval<Function>()))
    {
      return executor_.execute(std::forward<Function>(f));
    }

    template <typename Function>
    auto twoway_execute(Function&& f) const
      -> decltype(declval<typename conditional<true,
          Executor, Function>::type>().twoway_execute(declval<Function>()))
    {
      return executor_.execute(std::forward<Function>(f));
    }

    friend constexpr bool operator==(
        const adapter& a, const adapter& b) noexcept
    {
      return a.executor_ == b.executor_;
    }

    friend constexpr bool operator!=(
        const adapter& a, const adapter& b) noexcept
    {
      return a.executor_ != b.executor_;
    }

  private:
    template <typename> friend class adapter;
    Executor executor_;
  };

  template <typename Executor>
  friend adapter<Executor> require(Executor ex, allowed_t)
  {
    return adapter<Executor>(std::move(ex));
  }
};

constexpr blocking_adaptation_t blocking_adaptation{};
inline constexpr blocking_adaptation_t::disallowed_t blocking_adaptation_t::disallowed;
inline constexpr blocking_adaptation_t::allowed_t blocking_adaptation_t::allowed;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_BLOCKING_ADAPTATION_HPP
