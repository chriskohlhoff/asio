//
// echo_server_2.cpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio/ts/net.hpp>
#include <asio/use_fiber.hpp>

namespace ctx = boost::context;
using asio::ip::tcp;

asio::io_context io_ctx;

ctx::fiber listener(ctx::fiber f)
{
  auto use_fiber = asio::use_fiber(f);

  tcp::acceptor acceptor(io_ctx, {tcp::v4(), 54321});
  for (;;)
  {
    if (auto [_, socket] = acceptor.async_accept(use_fiber); socket.is_open())
    {
      for (;;)
      {
        char data[1024];
        auto [e1, n] = socket.async_read_some(asio::buffer(data), use_fiber);
        if (e1) break;
        auto [e2, _] = asio::async_write(socket, asio::buffer(data, n), use_fiber);
        if (e2) break;
      }
    }
  }
}

int main()
{
  ctx::fiber(listener).resume();
  io_ctx.run();
}
