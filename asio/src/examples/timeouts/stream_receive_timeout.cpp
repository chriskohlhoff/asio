//
// stream_receive_timeout.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
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

class stream_handler
{
public:
  stream_handler(io_service& ios)
    : io_service_(ios),
      timer_(ios),
      acceptor_(ios, tcp::endpoint(tcp::v4(), 32123)),
      socket_(ios)
  {
    acceptor_.async_accept(socket_,
        boost::bind(&stream_handler::handle_accept, this,
          asio::placeholders::error));
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

      socket_.async_read_some(asio::buffer(buf_, sizeof(buf_)),
          boost::bind(&stream_handler::handle_recv, this,
            asio::placeholders::error));
      timer_.expires_from_now(boost::posix_time::seconds(5));
      timer_.async_wait(boost::bind(&stream_handler::close, this));
    }
  }

  void handle_recv(const asio::error_code& err)
  {
    if (err)
    {
      std::cout << "Receive error: " << err.message() << "\n";
    }
    else
    {
      std::cout << "Successful receive\n";
    }
  }

  void close()
  {
    socket_.close();
  }

private:
  io_service& io_service_;
  deadline_timer timer_;
  tcp::acceptor acceptor_;
  tcp::socket socket_;
  char buf_[1024];
};

void connect_handler()
{
  std::cout << "Successful connect\n";
}

int main()
{
  try
  {
    io_service ios;

    stream_handler sh(ios);

    tcp::socket s(ios);
    s.async_connect(
        tcp::endpoint(asio::ip::address_v4::loopback(), 32123),
        boost::bind(connect_handler));

    ios.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
