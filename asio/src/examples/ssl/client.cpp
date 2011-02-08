//
// client.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include "asio.hpp"
#include "asio/ssl.hpp"

enum { max_length = 1024 };

class client
{
public:
  client(asio::io_service& io_service, asio::ssl::context& context,
      asio::ip::tcp::resolver::iterator endpoint_iterator)
    : socket_(io_service, context)
  {
    asio::ip::tcp::endpoint endpoint = *endpoint_iterator;
    socket_.lowest_layer().async_connect(endpoint,
        boost::bind(&client::handle_connect, this,
          asio::placeholders::error, ++endpoint_iterator));
  }

  void handle_connect(const asio::error_code& error,
      asio::ip::tcp::resolver::iterator endpoint_iterator)
  {
    if (!error)
    {
      socket_.async_handshake(asio::ssl::stream_base::client,
          boost::bind(&client::handle_handshake, this,
            asio::placeholders::error));
    }
    else if (endpoint_iterator != asio::ip::tcp::resolver::iterator())
    {
      socket_.lowest_layer().close();
      asio::ip::tcp::endpoint endpoint = *endpoint_iterator;
      socket_.lowest_layer().async_connect(endpoint,
          boost::bind(&client::handle_connect, this,
            asio::placeholders::error, ++endpoint_iterator));
    }
    else
    {
      std::cout << "Connect failed: " << error << "\n";
    }
  }

  void handle_handshake(const asio::error_code& error)
  {
    if (!error)
    {
      std::cout << "Enter message: ";
      std::cin.getline(request_, max_length);
      size_t request_length = strlen(request_);

      asio::async_write(socket_,
          asio::buffer(request_, request_length),
          boost::bind(&client::handle_write, this,
            asio::placeholders::error,
            asio::placeholders::bytes_transferred));
    }
    else
    {
      std::cout << "Handshake failed: " << error << "\n";
    }
  }

  void handle_write(const asio::error_code& error,
      size_t bytes_transferred)
  {
    if (!error)
    {
      asio::async_read(socket_,
          asio::buffer(reply_, bytes_transferred),
          boost::bind(&client::handle_read, this,
            asio::placeholders::error,
            asio::placeholders::bytes_transferred));
    }
    else
    {
      std::cout << "Write failed: " << error << "\n";
    }
  }

  void handle_read(const asio::error_code& error,
      size_t bytes_transferred)
  {
    if (!error)
    {
      std::cout << "Reply: ";
      std::cout.write(reply_, bytes_transferred);
      std::cout << "\n";
    }
    else
    {
      std::cout << "Read failed: " << error << "\n";
    }
  }

private:
  asio::ssl::stream<asio::ip::tcp::socket> socket_;
  char request_[max_length];
  char reply_[max_length];
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 3)
    {
      std::cerr << "Usage: client <host> <port>\n";
      return 1;
    }

    asio::io_service io_service;

    asio::ip::tcp::resolver resolver(io_service);
    asio::ip::tcp::resolver::query query(argv[1], argv[2]);
    asio::ip::tcp::resolver::iterator iterator = resolver.resolve(query);

    asio::ssl::context ctx(io_service, asio::ssl::context::sslv23);
    ctx.set_verify_mode(asio::ssl::context::verify_peer);
    ctx.load_verify_file("ca.pem");

    client c(io_service, ctx, iterator);

    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
