//
// blocking_udp_client.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "asio/deadline_timer.hpp"
#include "asio/io_service.hpp"
#include "asio/ip/udp.hpp"
#include <cstdlib>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>

using asio::ip::udp;

//----------------------------------------------------------------------

class client
{
public:
  client(const udp::endpoint& listen_endpoint)
    : socket_(io_service_, listen_endpoint),
      timer_(io_service_)
  {
    timer_.async_wait(boost::bind(&client::handle_timeout, this));
  }

  std::size_t receive(const asio::mutable_buffer& buffer,
      boost::posix_time::time_duration timeout, asio::error_code& ec)
  {
    timer_.expires_from_now(timeout);

    ec = asio::error::would_block;
    std::size_t length = 0;

    socket_.async_receive(asio::buffer(buffer),
        boost::bind(&client::handle_receive, _1, _2, &ec, &length));

    do io_service_.run_one(); while (ec == asio::error::would_block);

    return length;
  }

private:
  void handle_timeout()
  {
    if (timer_.expires_from_now() <= boost::posix_time::seconds(0))
    {
      socket_.cancel();
      timer_.expires_at(boost::posix_time::pos_infin);
    }

    timer_.async_wait(boost::bind(&client::handle_timeout, this));
  }

  static void handle_receive(
      const asio::error_code& ec, std::size_t length,
      asio::error_code* out_ec, std::size_t* out_length)
  {
    *out_ec = ec;
    *out_length = length;
  }

private:
  asio::io_service io_service_;
  udp::socket socket_;
  asio::deadline_timer timer_;
};

//----------------------------------------------------------------------

int main(int argc, char* argv[])
{
  try
  {
    using namespace std; // For atoi.

    if (argc != 3)
    {
      std::cerr << "Usage: blocking_udp_timeout <listen_addr> <listen_port>\n";
      return 1;
    }

    udp::endpoint listen_endpoint(
        asio::ip::address::from_string(argv[1]),
        std::atoi(argv[2]));

    client c(listen_endpoint);

    for (;;)
    {
      char data[1024];
      asio::error_code ec;
      std::size_t n = c.receive(asio::buffer(data),
          boost::posix_time::seconds(10), ec);

      if (ec)
      {
        std::cout << "Receive error: " << ec.message() << "\n"; 
      }
      else
      {
        std::cout << "Received: ";
        std::cout.write(data, n);
        std::cout << "\n";
      }
    }
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
