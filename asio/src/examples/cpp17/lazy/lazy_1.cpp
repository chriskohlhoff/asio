//
// lazy_1.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio/ts/net.hpp>
#include <asio/lazy.hpp>
#include <iostream>

int main()
{
  asio::io_context ctx;
  asio::steady_timer timer(ctx);
  timer.expires_after(std::chrono::seconds(1));
  auto x = timer.async_wait(asio::lazy);
  x([](auto){ std::cout << "timer done\n"; });
  ctx.run();
}
