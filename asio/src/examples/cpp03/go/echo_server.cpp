//
// echo_server.cpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio/deadline_timer.hpp>
#include <asio/go.hpp>
#include <asio/io_service.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/write.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <iostream>

#include <asio/yield.hpp>

using asio::ip::tcp;

class session : public boost::enable_shared_from_this<session>
{
public:
  explicit session(asio::io_service& io_service)
    : strand_(io_service),
      socket_(io_service),
      timer_(io_service)
  {
  }

  tcp::socket& socket()
  {
    return socket_;
  }

  void go()
  {
    asio::go(strand_,
        boost::bind(&session::echo,
          shared_from_this(), _1));
    asio::go(strand_,
        boost::bind(&session::timeout,
          shared_from_this(), _1));
  }

private:
  char data_[128];
  std::size_t n_;

  void echo(asio::stackless_context ctx)
  {
    try
    {
      reenter (ctx)
      {
        for (;;)
        {
          timer_.expires_from_now(boost::posix_time::seconds(10));
          await n_ = socket_.async_read_some(asio::buffer(data_), ctx);
          yield asio::async_write(socket_, asio::buffer(data_, n_), ctx);
        }
      }
    }
    catch (std::exception& e)
    {
      socket_.close();
      timer_.cancel();
    }
  }

  asio::error_code ignored_ec_;

  void timeout(asio::stackless_context ctx)
  {
    reenter (ctx)
    {
      while (socket_.is_open())
      {
        yield timer_.async_wait(ctx[ignored_ec_]);
        if (timer_.expires_from_now() <= boost::posix_time::seconds(0))
          socket_.close();
      }
    }
  }

  asio::io_service::strand strand_;
  tcp::socket socket_;
  asio::deadline_timer timer_;
};

struct do_accept
{
  tcp::acceptor& acceptor_;
  asio::error_code ec_;
  boost::shared_ptr<session> new_session_;

  explicit do_accept(tcp::acceptor& acceptor)
    : acceptor_(acceptor)
  {
  }

  void operator()(asio::stackless_context ctx)
  {
    reenter (ctx)
    {
      for (;;)
      {
        new_session_.reset(new session(acceptor_.get_io_service()));
        yield acceptor_.async_accept(new_session_->socket(), ctx[ec_]);
        if (!ec_) new_session_->go();
      }
    }
  }
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
    asio::go(io_service, do_accept(acceptor));
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
