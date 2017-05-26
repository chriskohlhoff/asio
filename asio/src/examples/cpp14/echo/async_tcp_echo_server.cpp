//
// async_tcp_echo_server.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2016 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <asio/ts/buffer.hpp>
#include <asio/ts/internet.hpp>

using asio::ip::tcp;

class session
{
public:
  session(tcp::socket &&socket)
    : socket_(std::move(socket))
  {
  }

  void start(std::unique_ptr<session> &&self)
  {
    do_read(std::move(self));
  }

private:
  void do_read(std::unique_ptr<session> &&self)
  {
    socket_.async_read_some(asio::buffer(data_, max_length),
        [this, s = std::move(self)](std::error_code ec, std::size_t length) mutable
        {
          if (!ec)
          {
            do_write(std::move(s), length);
          }
        });
  }

  void do_write(std::unique_ptr<session> &&self, std::size_t length)
  {
    asio::async_write(socket_, asio::buffer(data_, length),
        [this, s = std::move(self)](std::error_code ec, std::size_t /*length*/) mutable
        {
          if (!ec)
          {
            do_read(std::move(s));
          }
        });
  }

  tcp::socket socket_;
  enum { max_length = 1024 };
  char data_[max_length];
};

class server
{
public:
  server(asio::io_context& io_context, short port)
    : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)),
      socket_(io_context)
  {
    do_accept();
  }

private:
  void do_accept()
  {
    acceptor_.async_accept(socket_,
        [this](std::error_code ec)
        {
          if (!ec)
          {
            auto s = std::make_unique<session>(std::move(socket_));
            s->start(std::move(s));
          }

          do_accept();
        });
  }

  tcp::acceptor acceptor_;
  tcp::socket socket_;
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: async_tcp_echo_server <port>\n";
      return 1;
    }

    asio::io_context io_context;

    server s(io_context, std::atoi(argv[1]));

    io_context.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
