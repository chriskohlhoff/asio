//
// accept_timeout.cpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "asio.hpp"
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>

using namespace asio;
using asio::ip::tcp;

class accept_handler
{
public:
  accept_handler(io_service& ios)
    : io_service_(ios),
      timer_(ios),
      acceptor_(ios, tcp::endpoint(tcp::v4(), 32123)),
      socket_(ios)
  {
    acceptor_.async_accept(socket_,
        boost::bind(&accept_handler::handle_accept, this,
          asio::placeholders::error));

    timer_.expires_from_now(boost::posix_time::seconds(5));
    timer_.async_wait(boost::bind(&accept_handler::close, this));
  }

  void handle_accept(const asio::error_code& err)
  {
    if (err)
    {
      std::cout << "Accept error: " << err.message() << "\n";
    }
    else
    {
      std::cout << "Successful accept\n";
    }
  }

  void close()
  {
    acceptor_.close();
  }

private:
  io_service& io_service_;
  deadline_timer timer_;
  tcp::acceptor acceptor_;
  tcp::socket socket_;
};

int main()
{
  try
  {
    io_service ios;
    accept_handler ah(ios);
    ios.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
