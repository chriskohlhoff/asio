//
// deferred_6.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio.hpp>
#include <asio/experimental/append.hpp>
#include <asio/experimental/deferred.hpp>
#include <iostream>

using asio::experimental::append;
using asio::experimental::deferred;

template <typename CompletionToken>
auto async_wait_twice(asio::steady_timer& timer, CompletionToken&& token)
{
  return deferred.values(asio::success, &timer)(
      deferred(
        [](asio::noerror, asio::steady_timer* timer)
        {
          timer->expires_after(std::chrono::seconds(1));
          return timer->async_wait(append(deferred, timer));
        }
      )
    )(
      deferred(
        [](auto ec, asio::steady_timer* timer)
        {
          std::cout << "first timer wait finished\n";
          timer->expires_after(std::chrono::seconds(1));
          return deferred.when(!ec)
            .then(timer->async_wait(deferred))
            .otherwise(deferred.values(ec));
        }
      )
    )(
      deferred(
        [](auto ec)
        {
          std::cout << "second timer wait finished\n";
          return deferred.when(!ec)
            .then(deferred.values(asio::success, 42))
            .otherwise(deferred.values(ec, 0));
        }
      )
    )(
      std::forward<CompletionToken>(token)
    );
}

int main()
{
  asio::io_context ctx;

  asio::steady_timer timer(ctx);
  timer.expires_after(std::chrono::seconds(1));

  async_wait_twice(
      timer,
      [](std::error_code, int result)
      {
        std::cout << "result is " << result << "\n";
      }
    );

  // Uncomment the following line to trigger an error in async_wait_twice.
  //timer.cancel();

  ctx.run();

  return 0;
}
