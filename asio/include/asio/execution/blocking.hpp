//
// execution/blocking.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXECUTION_BLOCKING_HPP
#define ASIO_EXECUTION_BLOCKING_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <future>
#include "asio/detail/type_traits.hpp"
#include "asio/execution/blocking_adaptation.hpp"
#include "asio/execution/detail/enumeration.hpp"
#include "asio/execution/detail/enumerator_adapter.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

struct blocking_t :
  detail::enumeration<blocking_t, 3>
{
  using detail::enumeration<blocking_t, 3>::enumeration;

  using possibly_t = enumerator<0>;
  using always_t = enumerator<1>;
  using never_t = enumerator<2>;

  static constexpr possibly_t possibly{};
  static constexpr always_t always{};
  static constexpr never_t never{};

private:
  template <typename Executor>
  class adapter :
    public detail::enumerator_adapter<adapter,
      Executor, blocking_t, always_t>
  {
  public:
    using detail::enumerator_adapter<adapter, Executor,
      blocking_t, always_t>::enumerator_adapter;

    template <typename Function>
    auto execute(Function&& f) const
      -> decltype(declval<typename conditional<true,
          Executor, Function>::type>().execute(declval<Function>()))
    {
      std::promise<void> promise;
      std::future<void> future = promise.get_future();
      this->executor_.execute(
          [f = std::move(f), promise = std::move(promise)]() mutable
          {
            f();
          });
      future.wait();
    }

    template <typename Function>
    auto twoway_execute(Function&& f) const
      -> decltype(declval<typename conditional<true,
          Executor, Function>::type>().twoway_execute(declval<Function>()))
    {
      auto future = this->executor_.twoway_execute(std::forward<Function>(f));
      future.wait();
      return future;
    }
  };

public:
  template <typename Executor,
    typename = typename enable_if<
      blocking_adaptation_t::static_query_v<Executor>
        == blocking_adaptation.allowed
    >::type>
  friend adapter<Executor> require(Executor ex, always_t)
  {
    return adapter<Executor>(std::move(ex));
  }
};

constexpr blocking_t blocking{};
inline constexpr blocking_t::possibly_t blocking_t::possibly;
inline constexpr blocking_t::always_t blocking_t::always;
inline constexpr blocking_t::never_t blocking_t::never;

} // namespace execution
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXECUTION_BLOCKING_HPP
