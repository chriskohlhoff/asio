//
// experimental/linked_group.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_LINKED_GROUP_HPP
#define ASIO_EXPERIMENTAL_LINKED_GROUP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associator.hpp"
#include "asio/async_result.hpp"
#include "asio/error.hpp"
#include "asio/experimental/linked_continuation.hpp"
#include <tuple>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace detail {

template <std::size_t I, class CompletionHandler, class... Ops>
struct launch_linked_group
{
  CompletionHandler completion_handler;
  std::tuple<Ops...>& ops;
};

} // namespace detail

template <class... Ops>
class linked_group
{
public:
  static_assert(sizeof...(Ops) > 0);

  linked_group(Ops... ops)
    : ops_(std::move(ops)...)
  {
  }

  template <class CompletionToken>
  auto async_wait(CompletionToken&& token) &&
  {
    return asio::async_initiate<CompletionToken, void(std::error_code)>(
        [](auto completion_handler, std::tuple<Ops...>&& ops)
        {
          constexpr std::size_t last = sizeof...(Ops) - 1;
          std::move(std::get<last>(ops))(
              detail::launch_linked_group<last,
                decltype(completion_handler), Ops...>{
                  std::move(completion_handler), ops});
        }, token, std::move(ops_));
  }

private:
  std::tuple<Ops...> ops_;
};

template <class... Ops>
inline linked_group<Ops...> make_linked_group(Ops... ops)
{
  return linked_group<Ops...>(std::move(ops)...);
}

} // namespace experimental

template <std::size_t I, class CompletionHandler,
    class... Ops, class... Signatures>
struct async_result<
    experimental::detail::launch_linked_group<
      I, CompletionHandler, Ops...>, Signatures...>
{
  template <class Init, class... InitArgs>
  static void initiate(Init init,
      experimental::detail::launch_linked_group<
        I, CompletionHandler, Ops...> token,
      InitArgs... init_args)
  {
    if constexpr (I == 0)
    {
      std::move(std::get<0>(token.ops))(std::move(token.completion_handler));
    }
    else
    {
      constexpr std::size_t prev = I - 1;
      std::move(std::get<prev>(token.ops))(
          experimental::detail::launch_linked_group<prev,
            experimental::linked_continuation<
              CompletionHandler, Init, InitArgs...>, Ops...>{
                {std::move(token.completion_handler), std::move(init),
                  {std::move(init_args)...}}, token.ops});
    }
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_LINKED_GROUP_HPP
