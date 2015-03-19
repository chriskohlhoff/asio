//
// daytime_client.cpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <array>
#include <future>
#include <iostream>
#include <thread>
#include <asio/io_service.hpp>
#include <asio/ip/udp.hpp>
#include <asio/use_future.hpp>

using asio::ip::udp;

void get_daytime(asio::io_service& io_service, const char* hostname)
{
  try
  {
    udp::resolver resolver(io_service);

    std::future<udp::resolver::iterator> iter =
      resolver.async_resolve(
          {udp::v4(), hostname, "daytime"},
          asio::use_future);

    // The async_resolve operation above returns the endpoint iterator as a
    // future value that is not retrieved ...

    udp::socket socket(io_service, udp::v4());

    std::array<char, 1> send_buf  = {{ 0 }};
    std::future<std::size_t> send_length =
      socket.async_send_to(asio::buffer(send_buf),
          *iter.get(), // ... until here. This call may block.
          asio::use_future);

    // Do other things here while the send completes.

    send_length.get(); // Blocks until the send is complete. Throws any errors.

    std::array<char, 128> recv_buf;
    udp::endpoint sender_endpoint;
    std::future<std::size_t> recv_length =
      socket.async_receive_from(
          asio::buffer(recv_buf),
          sender_endpoint,
          asio::use_future);

    // Do other things here while the receive completes.

    std::cout.write(
        recv_buf.data(),
        recv_length.get()); // Blocks until receive is complete.
  }
  catch (std::system_error& e)
  {
    std::cerr << e.what() << std::endl;
  }
}

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: daytime_client <host>" << std::endl;
      return 1;
    }

    // We run the io_service off in its own thread so that it operates
    // completely asynchronously with respect to the rest of the program.
    asio::io_service io_service;
    asio::io_service::work work(io_service);
    std::thread thread([&io_service](){ io_service.run(); });

    get_daytime(io_service, argv[1]);

    io_service.stop();
    thread.join();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
