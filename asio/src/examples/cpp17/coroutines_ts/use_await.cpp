//
// use_await.cpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <atomic>
#include <experimental/coroutine>
#include <new>
#include <tuple>
#include <type_traits>

//------------------------------------------------------------------------------

class simple_coro
{
public:
  simple_coro(const simple_coro&) = delete;
  simple_coro& operator=(const simple_coro&) = delete;

  explicit simple_coro(std::experimental::coroutine_handle<> coro)
    : coro_(coro)
  {
  }

  simple_coro(simple_coro&& other) noexcept
    : coro_(std::exchange(other.coro_, nullptr))
  {
  }

  simple_coro& operator=(simple_coro&& other) noexcept
  {
    simple_coro tmp(std::move(other));
    std::swap(coro_, tmp.coro_);
    return *this;
  }

  ~simple_coro()
  {
    if (coro_)
      coro_.destroy();
  }

  void resume()
  {
    coro_.resume();
  }

private:
  std::experimental::coroutine_handle<> coro_;
};

struct simple_promise
{
  simple_coro get_return_object()
  {
    return simple_coro{std::experimental::coroutine_handle<simple_promise>::from_promise(*this)};
  };

  auto initial_suspend()
  {
    return std::experimental::suspend_always();
  }

  auto final_suspend()
  {
    return std::experimental::suspend_always();
  }

  void return_void()
  {
  }

  void unhandled_exception()
  {
    std::terminate();
  }
};

namespace std { namespace experimental {

template <typename... Args>
struct coroutine_traits<simple_coro, Args...>
{
  using promise_type = simple_promise;
};

}} // namespace std::experimental

//------------------------------------------------------------------------------

#include <asio/ts/net.hpp>
#include <asio/signal_set.hpp>
#include <asio/use_await.hpp>
#include <list>
#include <iostream>

using asio::ip::tcp;
using asio::use_await;

simple_coro listener(tcp::acceptor& acceptor)
{
  for (;;)
  {
    if (auto [_, socket] = co_await acceptor.async_accept(use_await); socket.is_open())
    {
      for (;;)
      {
        char data[1024];
        auto [e1, n] = co_await socket.async_read_some(asio::buffer(data), use_await);
        if (e1) break;
        auto [e2, _] = co_await asio::async_write(socket, asio::buffer(data, n), use_await);
        if (e2) break;
      }
    }
  }
}

simple_coro stopper(asio::io_context& ctx)
{
  asio::signal_set signals(ctx, SIGINT);
  auto [_, n] = co_await signals.async_wait(use_await);
  std::cout << "got sig " << n << "\n";
  ctx.stop();
}

int main()
{
  asio::io_context ctx(1);
  std::list<simple_coro> coros;
  tcp::acceptor acceptor(ctx, {tcp::v4(), 54321});
  for (std::size_t i = 0; i < 2; ++i)
    coros.push_back(listener(acceptor));
  coros.push_back(stopper(ctx));
  for (auto& coro : coros)
    coro.resume();
  ctx.run();
}
