//
// echo_server.cpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio/await.hpp>
#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/signal_set.hpp>
#include <asio/write.hpp>
#include <cstdio>

using asio::ip::tcp;

typedef asio::basic_unsynchronized_await_context<
  asio::io_context::executor_type> await_context;

asio::awaitable<void> echo_once(await_context ctx, tcp::socket& socket)
{
  char data[128];
  std::size_t n = co_await socket.async_read_some(asio::buffer(data), ctx);
  n = co_await async_write(socket, asio::buffer(data, n), ctx);
}

asio::awaitable<void> echo(await_context ctx, tcp::socket socket)
{
  try
  {
    for (;;)
    {
      co_await echo_once(ctx, socket);
    }
  }
  catch (std::exception& e)
  {
    std::printf("echo Exception: %s\n", e.what());
  }
}

asio::awaitable<void> listener(await_context ctx)
{
  tcp::acceptor acceptor(ctx.get_executor().context(), {tcp::v4(), 55555});
  for (;;)
    asio::spawn(echo, ctx, co_await acceptor.async_accept(ctx));
}

int main()
{
  try
  {
    asio::io_context io_context;
    asio::signal_set signals(io_context, SIGINT, SIGTERM);
    signals.async_wait([&](auto, auto){ io_context.stop(); });
    asio::spawn(listener, io_context);
    io_context.run();
  }
  catch (std::exception& e)
  {
    std::printf("Exception: %s\n", e.what());
  }
}
