//
// blocking_udp.cpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "asio/deadline_timer.hpp"
#include "asio/io_service.hpp"
#include "asio/ip/multicast.hpp"
#include "asio/ip/udp.hpp"
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>

using asio::ip::udp;
using asio::ip::multicast::join_group;

const short multicast_port = 30001;

class receiver
{
public:
  receiver(const asio::ip::address& listen_address,
      const asio::ip::address& multicast_address)
    : socket_(io_service_, udp::endpoint(listen_address, multicast_port)),
      timer_(io_service_)
  {
    socket_.set_option(join_group(multicast_address));

    timer_.async_wait(boost::bind(&receiver::handle_timeout, this));
  }

  std::size_t receive(const asio::mutable_buffer& buffer,
      boost::posix_time::time_duration timeout, asio::error_code& ec)
  {
    timer_.expires_from_now(timeout);

    ec = asio::error::would_block;
    std::size_t length = 0;

    socket_.async_receive(asio::buffer(buffer),
        boost::bind(&receiver::handle_receive, _1, _2, &ec, &length));

    do io_service_.run_one() while (ec == asio::error::would_block);

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

    timer_.async_wait(boost::bind(&receiver::handle_timeout, this));
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

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 3)
    {
      std::cerr << "Usage: blocking_udp_timeout <listen_addr> <mcast_addr>\n";
      std::cerr << "  For IPv4, try:\n";
      std::cerr << "    receiver 0.0.0.0 239.255.0.1\n";
      std::cerr << "  For IPv6, try:\n";
      std::cerr << "    receiver 0::0 ff31::8000:1234\n";
      return 1;
    }

    receiver r(asio::ip::address::from_string(argv[1]),
        asio::ip::address::from_string(argv[2]));

    for (;;)
    {
      char data[1024];
      asio::error_code ec;
      std::size_t n = r.receive(asio::buffer(data),
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
