//
// eager_4.cpp
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

  asio::steady_timer timer(ctx);
  timer.expires_after(std::chrono::seconds(5));

  auto eager_op = timer.async_wait(eager);

  // A cancellation slot may be bound to the handler that awaits the result,
  // as for any other asynchronous operation.
  asio::cancellation_signal sig;

  std::move(eager_op)(
      asio::bind_cancellation_slot(sig.slot(),
        [](std::error_code ec)
        {
          std::cout << "timer wait finished: " << ec.message() << "\n";
        }
      )
    );

  // The signal is emitted from within the io_context, where the wait was
  // launched.
  asio::steady_timer trigger(ctx);
  trigger.expires_after(std::chrono::milliseconds(100));
  trigger.async_wait(
      [&](std::error_code)
      {
        std::cout << "requesting cancellation\n";
        sig.emit(asio::cancellation_type::all);
      }
    );

  ctx.run();

  return 0;
}
