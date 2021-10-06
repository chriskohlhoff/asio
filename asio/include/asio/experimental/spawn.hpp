//
// experimental/spawn.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_SPAWN_HPP
#define ASIO_EXPERIMENTAL_SPAWN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/spawn.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {

template <typename Executor, typename Function, typename CompletionToken>
auto spawn(Executor e, Function f, CompletionToken&& token)
{
  return asio::async_initiate<CompletionToken, void()>(
      [](auto handler, Executor e, Function f)
      {
        asio::spawn(e,
            [e, f = std::move(f), handler = std::move(handler)](
              asio::yield_context yield) mutable
            {
              std::move(f)(yield);
              std::move(handler)();
            });
      }, token, std::move(e), std::move(f));
}

} // namespace experimental
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_ESSPAWN_HPP
