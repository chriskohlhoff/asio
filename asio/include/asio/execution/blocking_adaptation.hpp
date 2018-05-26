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
#include "asio/detail/type_traits.hpp"
#include "asio/execution/detail/enumeration.hpp"
#include "asio/execution/detail/enumerator_adapter.hpp"

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

private:
  template <typename Executor>
  class adapter :
    public detail::enumerator_adapter<adapter,
      Executor, blocking_adaptation_t, allowed_t>
  {
  public:
    using detail::enumerator_adapter<adapter, Executor,
      blocking_adaptation_t, allowed_t>::enumerator_adapter;

    template <typename Function>
    auto execute(Function&& f) const
      -> decltype(declval<typename conditional<true,
          Executor, Function>::type>().execute(declval<Function>()))
    {
      return this->executor_.execute(std::forward<Function>(f));
    }

    template <typename Function>
    auto twoway_execute(Function&& f) const
      -> decltype(declval<typename conditional<true,
          Executor, Function>::type>().twoway_execute(declval<Function>()))
    {
      return this->executor_.twoway_execute(std::forward<Function>(f));
    }
  };

public:
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
