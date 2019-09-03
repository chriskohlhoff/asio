//
// lazy_2.cpp
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

template <asio::completion_token_for<void(int)> CompletionToken>
auto async_foo(int i, CompletionToken&& token)
{
  return asio::async_initiate<CompletionToken, void(int)>(
      [](auto&& handler, int i)
      {
        auto ex = asio::get_associated_executor(handler);
        auto alloc = asio::get_associated_allocator(handler);
        int result = i * 2;
        ex.post(
            [handler = std::move(handler), result]() mutable
            {
              handler(result);
            }, alloc);
      }, token, i);
}

int main()
{
  auto x = async_foo(21, asio::lazy);
  x([](auto result){ std::cout << "result is " << result << "\n"; });
  asio::system_executor().context().join();
}
