//
// eager_3.cpp
// ~~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio.hpp>
#include <iostream>

using asio::eager;

int main()
{
  asio::io_context ctx;

  asio::steady_timer timer1(ctx);
  asio::steady_timer timer2(ctx);

  timer1.expires_after(std::chrono::milliseconds(100));
  timer2.expires_after(std::chrono::seconds(5));

  auto eager_op1 = timer1.async_wait(eager);
  auto eager_op2 = timer2.async_wait(eager);

  // The first result is awaited before deciding what to do with the second.
  std::move(eager_op1)(
      [&](std::error_code ec)
      {
        std::cout << "first timer wait finished: " << ec.message() << "\n";

        // Discarding the handle would detach, so cancel it instead.
        eager_op2.cancel();

        std::move(eager_op2)(
            [](std::error_code ec)
            {
              std::cout << "second timer wait finished: "
                << ec.message() << "\n";
            }
          );
      }
    );

  ctx.run();

  return 0;
}
