//
// echo_server.cpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio/go.hpp>
#include <asio/io_service.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/steady_timer.hpp>
#include <asio/write.hpp>
#include <iostream>
#include <memory>

#include <asio/yield.hpp>

using asio::ip::tcp;

class session : public std::enable_shared_from_this<session>
{
public:
  explicit session(tcp::socket socket)
    : socket_(std::move(socket)),
      timer_(socket_.get_io_service()),
      strand_(socket_.get_io_service())
  {
  }

  void go()
  {
    auto self(shared_from_this());

    std::size_t n = 0;
    std::array<char, 128> data;

    asio::go(strand_,
        [this, self, n, data](asio::stackless_context ctx) mutable
        {
          try
          {
            reenter (ctx)
            {
              for (;;)
              {
                timer_.expires_from_now(std::chrono::seconds(10));
                await n = socket_.async_read_some(asio::buffer(data), ctx);
                yield asio::async_write(socket_, asio::buffer(data, n), ctx);
              }
            }
          }
          catch (std::exception& e)
          {
            socket_.close();
            timer_.cancel();
          }
        });

    asio::error_code ignored_ec;

    asio::go(strand_,
        [this, self, ignored_ec](asio::stackless_context ctx) mutable
        {
          reenter (ctx)
          {
            while (socket_.is_open())
            {
              yield timer_.async_wait(ctx[ignored_ec]);
              if (timer_.expires_from_now() <= std::chrono::seconds(0))
                socket_.close();
            }
          }
        });
  }

private:
  tcp::socket socket_;
  asio::steady_timer timer_;
  asio::io_service::strand strand_;
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: echo_server <port>\n";
      return 1;
    }

    asio::io_service io_service;
    tcp::endpoint endpoint(tcp::v4(), std::atoi(argv[1]));
    tcp::acceptor acceptor(io_service, endpoint);
    tcp::socket socket(io_service);
    asio::error_code ec;

    asio::go(io_service,
        [&](asio::stackless_context ctx)
        {
          reenter (ctx)
          {
            for (;;)
            {
              yield acceptor.async_accept(socket, ctx[ec]);
              if (!ec) std::make_shared<session>(std::move(socket))->go();
            }
          }
        });

    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
