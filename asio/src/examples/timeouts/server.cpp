//
// server.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <set>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include "asio/deadline_timer.hpp"
#include "asio/io_service.hpp"
#include "asio/ip/tcp.hpp"
#include "asio/ip/udp.hpp"
#include "asio/read_until.hpp"
#include "asio/streambuf.hpp"
#include "asio/write.hpp"

using asio::ip::tcp;
using asio::ip::udp;

//----------------------------------------------------------------------

class subscriber
{
public:
  virtual ~subscriber() {}
  virtual void deliver(const std::string& msg) = 0;
};

typedef boost::shared_ptr<subscriber> subscriber_ptr;

//----------------------------------------------------------------------

class channel
{
public:
  void join(subscriber_ptr subscriber)
  {
    subscribers_.insert(subscriber);
  }

  void leave(subscriber_ptr subscriber)
  {
    subscribers_.erase(subscriber);
  }

  void deliver(const std::string& msg)
  {
    std::for_each(subscribers_.begin(), subscribers_.end(),
        boost::bind(&subscriber::deliver, _1, boost::ref(msg)));
  }

private:
  std::set<subscriber_ptr> subscribers_;
};

//----------------------------------------------------------------------

class tcp_session
  : public subscriber,
    public boost::enable_shared_from_this<tcp_session>
{
public:
  tcp_session(asio::io_service& io_service, channel& ch)
    : channel_(ch),
      socket_(io_service),
      input_timer_(io_service),
      non_empty_output_queue_(io_service),
      output_timer_(io_service)
  {
    input_timer_.expires_at(boost::posix_time::pos_infin);
    non_empty_output_queue_.expires_at(boost::posix_time::pos_infin);
    output_timer_.expires_at(boost::posix_time::pos_infin);
  }

  tcp::socket& socket()
  {
    return socket_;
  }

  void start()
  {
    channel_.join(shared_from_this());

    start_read();

    input_timer_.async_wait(
        boost::bind(&tcp_session::handle_timeout,
        shared_from_this(), &input_timer_));

    await_output();

    output_timer_.async_wait(
        boost::bind(&tcp_session::handle_timeout,
        shared_from_this(), &output_timer_));
  }

private:
  void stop()
  {
    channel_.leave(shared_from_this());

    socket_.close();
    input_timer_.cancel();
    non_empty_output_queue_.cancel();
    output_timer_.cancel();
  }

  bool stopped() const
  {
    return !socket_.is_open();
  }

  void deliver(const std::string& msg)
  {
    output_queue_.push_back(msg + "\n");

    non_empty_output_queue_.expires_at(boost::posix_time::neg_infin);
  }

  void start_read()
  {
    // Set a timeout for the read operation.
    input_timer_.expires_from_now(boost::posix_time::seconds(30));

    asio::async_read_until(socket_, input_buffer_, '\n',
        boost::bind(&tcp_session::handle_read, shared_from_this(), _1));
  }

  void handle_read(const asio::error_code& ec)
  {
    if (stopped())
      return;

    if (!ec)
    {
      std::string msg;
      std::istream is(&input_buffer_);
      std::getline(is, msg);

      if (!msg.empty())
      {
        channel_.deliver(msg);
      }
      else
      {
        // We received a heartbeat message from the client. If there's nothing
        // else being sent or ready to be sent, send a heartbeat right back.
        if (output_queue_.empty())
        {
          output_queue_.push_back("\n");
          non_empty_output_queue_.expires_at(boost::posix_time::neg_infin);
        }
      }

      start_read();
    }
    else
    {
      socket_.close();
    }
  }

  void await_output()
  {
    if (stopped())
      return;

    if (output_queue_.empty())
    {
      non_empty_output_queue_.expires_at(boost::posix_time::pos_infin);

      non_empty_output_queue_.async_wait(
          boost::bind(&tcp_session::await_output, shared_from_this()));
    }
    else
    {
      start_write();
    }
  }

  void start_write()
  {
    // Set a timeout for the write operation.
    output_timer_.expires_from_now(boost::posix_time::seconds(30));

    asio::async_write(socket_, asio::buffer(output_queue_.front()),
        boost::bind(&tcp_session::handle_write, shared_from_this(), _1));
  }

  void handle_write(const asio::error_code& ec)
  {
    if (stopped())
      return;

    if (!ec)
    {
      output_queue_.pop_front();

      await_output();
    }
    else
    {
      stop();
    }
  }

  void handle_timeout(asio::deadline_timer* timer)
  {
    if (stopped())
      return;

    // If this timeout actor is being woken up due to timer modification, then
    // the current expiry time will be in the future. We check the expiry time
    // to determine whether this is a genuine timeout.
    if (timer->expires_from_now() <= boost::posix_time::seconds(0))
    {
      // The timer has truly expired. The session will be terminated.
      stop();
    }
    else
    {
      // Put the timeout actor back to sleep.
      timer->async_wait(
          boost::bind(&tcp_session::handle_timeout,
          shared_from_this(), timer));
    }
  }

  channel& channel_;
  tcp::socket socket_;
  asio::streambuf input_buffer_;
  asio::deadline_timer input_timer_;
  std::deque<std::string> output_queue_;
  asio::deadline_timer non_empty_output_queue_;
  asio::deadline_timer output_timer_;
};

typedef boost::shared_ptr<tcp_session> tcp_session_ptr;

//----------------------------------------------------------------------

class udp_broadcaster
  : public subscriber
{
public:
  udp_broadcaster(asio::io_service& io_service,
      const udp::endpoint& broadcast_endpoint)
    : socket_(io_service)
  {
    socket_.connect(broadcast_endpoint);
  }

private:
  void deliver(const std::string& msg)
  {
    asio::error_code ignored_ec;
    socket_.send(asio::buffer(msg), 0, ignored_ec);
  }

  udp::socket socket_;
};

//----------------------------------------------------------------------

class server
{
public:
  server(asio::io_service& io_service,
      const tcp::endpoint& listen_endpoint,
      const udp::endpoint& broadcast_endpoint)
    : io_service_(io_service),
      acceptor_(io_service, listen_endpoint)
  {
    subscriber_ptr bc(new udp_broadcaster(io_service_, broadcast_endpoint));
    channel_.join(bc);

    tcp_session_ptr new_session(new tcp_session(io_service_, channel_));

    acceptor_.async_accept(new_session->socket(),
        boost::bind(&server::handle_accept, this, new_session, _1));
  }

  void handle_accept(tcp_session_ptr session,
      const asio::error_code& ec)
  {
    if (!ec)
    {
      session->start();

      tcp_session_ptr new_session(new tcp_session(io_service_, channel_));

      acceptor_.async_accept(new_session->socket(),
          boost::bind(&server::handle_accept, this, new_session, _1));
    }
  }

private:
  asio::io_service& io_service_;
  tcp::acceptor acceptor_;
  channel channel_;
};

//----------------------------------------------------------------------

int main(int argc, char* argv[])
{
  try
  {
    using namespace std; // For atoi.

    if (argc != 4)
    {
      std::cerr << "Usage: server <listen_port> <bcast_address> <bcast_port>\n";
      return 1;
    }

    asio::io_service io_service;

    tcp::endpoint listen_endpoint(tcp::v4(), atoi(argv[1]));

    udp::endpoint broadcast_endpoint(
        asio::ip::address::from_string(argv[2]), atoi(argv[3]));

    server s(io_service, listen_endpoint, broadcast_endpoint);

    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
