//
// exercise.cpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2026 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "asio.hpp"
#include <array>
#include <cstdio>
#include <thread>
#include <vector>

using asio::ip::tcp;

// A user composed operation that records its own handler tracking location and
// then delegates to asio::async_write. The async_write location slice opens and
// closes within this outer async_double_write location slice. The two nest on
// the same thread track.
template <class Stream, class Handler>
auto async_double_write(Stream& s, asio::const_buffer b, Handler&& h)
{
  return asio::async_compose<Handler, void(asio::error_code, std::size_t)>(
      [&s, b, started = false](auto& self,
          const asio::error_code& ec = asio::error_code(),
          std::size_t n = 0) mutable
      {
        ASIO_HANDLER_LOCATION((__FILE__, __LINE__, "async_double_write"));
        if (!started)
        {
          started = true;
          asio::async_write(s, b, std::move(self));
        }
        else
        {
          self.complete(ec, n);
        }
      }, h, s);
}

int main()
{
  asio::io_context ctx;

  tcp::acceptor acceptor(ctx, tcp::endpoint(tcp::v4(), 0));
  tcp::socket server(ctx);
  tcp::socket client(ctx);

  acceptor.async_accept(server, [](const asio::error_code&){});

  client.async_connect(acceptor.local_endpoint(),
      [&](const asio::error_code&)
      {
        static std::array<char, 5> out = {{'h', 'e', 'l', 'l', 'o'}};
        async_double_write(client, asio::buffer(out),
            [&](const asio::error_code&, std::size_t){ client.close(); });
      });

  static std::array<char, 5> in;
  asio::async_read(server, asio::buffer(in),
      [&](const asio::error_code&, std::size_t){ server.close(); });

  std::vector<std::thread> pool;
  for (int i = 0; i < 3; ++i)
    pool.emplace_back([&]{ ctx.run(); });
  for (std::size_t i = 0; i < pool.size(); ++i)
    pool[i].join();

  std::printf("exercise complete\n");
  return 0;
}
