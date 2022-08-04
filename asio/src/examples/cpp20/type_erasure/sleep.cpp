//
// sleep.cpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2022 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "sleep.hpp"
#include <asio/bind_executor.hpp>
#include <asio/steady_timer.hpp>
#include <memory>

void async_sleep_impl(
    asio::any_completion_handler<void(std::error_code)> handler,
    asio::any_io_executor ex, std::chrono::nanoseconds duration)
{
  auto timer = std::make_shared<asio::steady_timer>(ex, duration);
  auto handler_ex = asio::get_associated_executor(handler, ex);
  timer->async_wait(
      asio::bind_executor(handler_ex,
        [timer, handler = std::move(handler)](std::error_code ec) mutable
        {
          std::move(handler)(ec);
        }
      )
    );
}
