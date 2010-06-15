//
// blocking_tcp.cpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "asio/deadline_timer.hpp"
#include "asio/io_service.hpp"
#include "asio/ip/tcp.hpp"
#include "asio/read_until.hpp"
#include "asio/streambuf.hpp"
#include "asio/write.hpp"
#include "asio/thread.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>

using asio::ip::tcp;
using boost::lambda::bind;
using boost::lambda::var;
using boost::lambda::_1;

class line_based_client
{
public:
  line_based_client()
    : socket_(io_service_),
      timer_(io_service_)
  {
    timer_.async_wait(bind(&line_based_client::handle_timeout, this));
  }

  void connect(const std::string& host, const std::string& service,
      boost::posix_time::time_duration timeout)
  {
    timer_.expires_from_now(timeout);

    tcp::resolver::query query(host, service);
    tcp::resolver::iterator iter = tcp::resolver(io_service_).resolve(query);

    asio::error_code ec;

    for (; iter != tcp::resolver::iterator(); ++iter)
    {
      socket_.close();

      ec = asio::error::would_block;

      socket_.async_connect(iter->endpoint(), var(ec) = _1);

      do io_service_.run_one(); while (ec == asio::error::would_block);

      // Determine whether a connection was successfully established.
      if (!ec && socket_.is_open())
        return;
    }

    throw asio::system_error(ec ? ec : asio::error::host_not_found);
  }

  std::string read_line(boost::posix_time::time_duration timeout)
  {
    timer_.expires_from_now(timeout);

    asio::error_code ec = asio::error::would_block;

    // See blocking_udp example for how to use boost::bind rather than lambda in this scenario.
    asio::async_read_until(socket_, input_buffer_, "\r\n", var(ec) = _1);

    do io_service_.run_one(); while (ec == asio::error::would_block);

    if (ec)
      throw asio::system_error(ec);

    std::string line;
    std::istream is(&input_buffer_);
    std::getline(is, line, '\r');
    return line;
  }

  void write_line(const std::string& line,
      boost::posix_time::time_duration timeout)
  {
    std::string data = line + "\r\n";

    timer_.expires_from_now(timeout);

    asio::error_code ec = asio::error::would_block;

    asio::async_write(socket_, asio::buffer(data), var(ec) = _1);

    do io_service_.run_one(); while (ec == asio::error::would_block);

    if (ec)
      throw asio::system_error(ec);
  }

private:
  void handle_timeout()
  {
    if (timer_.expires_from_now() <= boost::posix_time::seconds(0))
    {
      socket_.close();
      timer_.expires_at(boost::posix_time::pos_infin);
    }

    timer_.async_wait(bind(&line_based_client::handle_timeout, this));
  }

  static void handle_io(const asio::error_code& ec,
      asio::error_code* out_ec)
  {
    *out_ec = ec;
  }

private:
  asio::io_service io_service_;
  tcp::socket socket_;
  asio::deadline_timer timer_;
  asio::streambuf input_buffer_;
};

int response_code(const std::string& line)
{
  using namespace std; // For atoi.
  return atoi(line.c_str());
}

int more_response(const std::string& line)
{
  return line.length() >= 4 && line[3] == '-';
}

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: blocking_tcp <host> <port>\n";
      return 1;
    }

    asio::error_code ec;
    line_based_client client;
    client.connect(argv[1], argv[2], boost::posix_time::seconds(10));

    std::string line = client.read_line(boost::posix_time::seconds(10));
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
