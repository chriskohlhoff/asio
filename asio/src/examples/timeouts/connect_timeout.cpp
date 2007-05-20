//
// connect_timeout.cpp
// ~~~~~~~~~~~~~~~~~~~
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

class connect_handler
{
public:
  connect_handler(io_service& ios)
    : io_service_(ios),
      timer_(ios),
      socket_(ios)
  {
    socket_.async_connect(
        tcp::endpoint(asio::ip::address_v4::loopback(), 32123),
        boost::bind(&connect_handler::handle_connect, this,
          asio::placeholders::error));

    timer_.expires_from_now(boost::posix_time::seconds(5));
    timer_.async_wait(boost::bind(&connect_handler::close, this));
  }

  void handle_connect(const asio::error_code& err)
  {
    if (err)
    {
      std::cout << "Connect error: " << err.message() << "\n";
    }
    else
    {
      std::cout << "Successful connection\n";
    }
  }

  void close()
  {
    socket_.close();
  }

private:
  io_service& io_service_;
  deadline_timer timer_;
  tcp::socket socket_;
};

int main()
{
  try
  {
    io_service ios;
    tcp::acceptor a(ios, tcp::endpoint(tcp::v4(), 32123), 1);

    // Make lots of connections so that at least some of them will block.
    connect_handler ch1(ios);
    connect_handler ch2(ios);
    connect_handler ch3(ios);
    connect_handler ch4(ios);
    connect_handler ch5(ios);
    connect_handler ch6(ios);
    connect_handler ch7(ios);
    connect_handler ch8(ios);
    connect_handler ch9(ios);

    ios.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
