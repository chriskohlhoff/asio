//
// tcp_server.cpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <asio/io_service.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/read.hpp>
#include <asio/write.hpp>
#include <boost/shared_ptr.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

using asio::ip::tcp;

#include "yield.hpp"

class tcp_server : coroutine
{
public:
  tcp_server(tcp::acceptor& acceptor, std::size_t bufsize) :
    acceptor_(acceptor),
    socket_(acceptor_.get_io_service()),
    buffer_(bufsize)
  {
  }

  void operator()(asio::error_code ec, std::size_t n = 0)
  {
    reenter (this) for (;;)
    {
      yield acceptor_.async_accept(socket_, ref(this));

      while (!ec)
      {
        yield asio::async_read(socket_,
            asio::buffer(buffer_), ref(this));

        if (!ec)
        {
          for (std::size_t i = 0; i < n; ++i) buffer_[i] = ~buffer_[i];

          yield asio::async_write(socket_,
              asio::buffer(buffer_), ref(this));
        }
      }

      socket_.close();
    }
  }

  struct ref
  {
    explicit ref(tcp_server* p) : p_(p) {}
    void operator()(asio::error_code ec, std::size_t n = 0) { (*p_)(ec, n); }
    private: tcp_server* p_;
  };

private:
  tcp::acceptor& acceptor_;
  tcp::socket socket_;
  std::vector<unsigned char> buffer_;
  tcp::endpoint sender_;
};

#include "unyield.hpp"

int main(int argc, char* argv[])
{
  if (argc != 4)
  {
    std::fprintf(stderr, "Usage: tcp_server <port> <nconns> <bufsize>\n");
    return 1;
  }

  asio::io_service io_service;
  unsigned short port = static_cast<unsigned short>(std::atoi(argv[1]));
  tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port));
  std::vector<boost::shared_ptr<tcp_server> > servers;

  int max_connections = std::atoi(argv[2]);
  std::size_t bufsize = std::atoi(argv[3]);
  for (int i = 0; i < max_connections; ++i)
  {
    boost::shared_ptr<tcp_server> s(new tcp_server(acceptor, bufsize));
    servers.push_back(s);
    (*s)(asio::error_code());
  }

  for (;;) io_service.poll();
}
