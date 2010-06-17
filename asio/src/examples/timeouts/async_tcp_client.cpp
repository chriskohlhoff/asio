//
// async_tcp_client.cpp
// ~~~~~~~~~~~~~~~~~~~~
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
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>

using asio::ip::tcp;

//
// This class consists of four asynchronous "actors":
//
//        Connect Actor                      Timeout Actor
//        ~~~~~~~~~~~~~                      ~~~~~~~~~~~~~
//
// +---------------+
// |               |
// | start_connect |<---+                  +----------------+
// |               |    |                  |                |     
// +---------------+    |                  | handle_timeout |<---+
//         |            |                  |                |    |
//         |    +----------------+         +----------------+    |
//         |    |                |                      |        |
//         +--->| handle_connect |                      +--------+
//              |                |
//              +----------------+
//                         :
//     Input Actor         :                Heartbeat Actor
//     ~~~~~~~~~~~         :                ~~~~~~~~~~~~~~~
//                         :    
//  +------------+         :         +-------------+
//  |            |<- - - - + - - - ->|             |
//  | start_read |                   | start_write |<----+
//  |            |<----+             |             |     |
//  +------------+     |             +-------------+     |
//         |           |                    |            |
//         |    +-------------+             |     +--------------+
//         |    |             |             |     |              |
//         +--->| handle_read |             +---->| handle_write |
//              |             |                   |              |
//              +-------------+                   +--------------+
//
// The Connect Actor performs connection establishment. It tries each endpoint
// in turn until a connection is established, or the available endpoints are
// exhausted. If a connection is successfully established, the Connect Actor
// forks into two new actors: the Input Actor and the Heartbeat Actor.
//
// The Input Actor reads messages from the socket. Messages are delimited by
// the newline character. The timeout for receiving a complete message is 30
// seconds.
//
// The Heartbeat Actor sends a heartbeat (a message that consists of a single
// newline character) every 10 seconds.
//
// The Timeout Actor is responsible for managing timeouts. When a timeout
// occurs it will close the socket. This will cause any pending asynchronous
// operations to complete with the operation_aborted error.
//
class client
{
public:
  client(asio::io_service& io_service)
    : stopped_(false),
      socket_(io_service),
      timeout_timer_(io_service),
      heartbeat_timer_(io_service)
  {
  }

  // Called by the user of the client class to initiate the connection process.
  // The endpoint iterator will have been obtained using a tcp::resolver.
  void start(tcp::resolver::iterator endpoint_iter)
  {
    // Start the connect actor.
    start_connect(endpoint_iter);

    // Start the timeout actor. You will note that we're not setting any
    // particular timeout here. Instead, the connect and input actors will
    // update the timeout prior to each asynchronous operation.
    timeout_timer_.async_wait(boost::bind(&client::handle_timeout, this));
  }

  // This function terminates all the actors to shut down the connection. It
  // may be called by the user of the client class, or by the class itself in
  // response to graceful termination or an unrecoverable error.
  void stop()
  {
    stopped_ = true;
    socket_.close();
    timeout_timer_.cancel();
    heartbeat_timer_.cancel();
  }

private:
  void start_connect(tcp::resolver::iterator endpoint_iter)
  {
    if (endpoint_iter != tcp::resolver::iterator())
    {
      std::cout << "Trying " << endpoint_iter->endpoint() << "...\n";

      // Set a timeout for the connect operation.
      timeout_timer_.expires_from_now(boost::posix_time::seconds(2));

      // Start the asynchronous connect operation.
      socket_.async_connect(endpoint_iter->endpoint(),
          boost::bind(&client::handle_connect,
            this, _1, endpoint_iter));
    }
    else
    {
      // There are no more endpoints to try. Shut down the client.
      stop();
    }
  }

  void handle_connect(const asio::error_code& ec,
      tcp::resolver::iterator endpoint_iter)
  {
    if (stopped_)
      return;

    // The async_connect() function automatically opens the socket at the start
    // of the asynchronous operation. If the socket is closed at this time then
    // the timeout handler must have run first.
    if (!socket_.is_open())
    {
      std::cout << "Connect timed out\n";

      // Try the next available endpoint.
      start_connect(++endpoint_iter);
    }

    // Check if the connect operation failed before the timeout period elapsed.
    else if (ec)
    {
      std::cout << "Connect error: " << ec.message() << "\n";

      // We need to close the socket used in the previous connection attempt
      // before starting a new one.
      socket_.close();

      // Try the next available endpoint.
      start_connect(++endpoint_iter);
    }

    // Otherwise we have successfully established a connection.
    else
    {
      std::cout << "Connected to " << endpoint_iter->endpoint() << "\n";

      // Start the input actor.
      start_read();

      // Start the heartbeat actor.
      start_write();
    }
  }

  void start_read()
  {
    // Set a timeout for the read operation.
    timeout_timer_.expires_from_now(boost::posix_time::seconds(30));

    asio::async_read_until(socket_, input_buffer_, '\n',
        boost::bind(&client::handle_read, this, _1));
  }

  void handle_read(const asio::error_code& ec)
  {
    if (stopped_)
      return;

    if (!ec)
    {
      std::string line;
      std::istream is(&input_buffer_);
      std::getline(is, line);

      if (!line.empty())
      {
        std::cout << "Received: " << line << "\n";
      }

      start_read();
    }
    else
    {
      std::cout << "Error on receive: " << ec.message() << "\n";

      stop();
    }
  }

  void start_write()
  {
    if (stopped_)
      return;

    asio::async_write(socket_, asio::buffer("\n", 1),
        boost::bind(&client::handle_write, this, _1));
  }

  void handle_write(const asio::error_code& ec)
  {
    if (stopped_)
      return;

    if (!ec)
    {
      heartbeat_timer_.expires_from_now(boost::posix_time::seconds(10));
      heartbeat_timer_.async_wait(boost::bind(&client::start_write, this));
    }
    else
    {
      std::cout << "Error on heartbeat: " << ec.message() << "\n";

      stop();
    }
  }

  void handle_timeout()
  {
    if (stopped_)
      return;

    // If the timeout actor is being woken up due to timer modification, then
    // the current expiry time will be in the future. We can use this
    // information to determine whether to close the socket and so cancel the
    // pending asynchronous protocol operations.
    if (timeout_timer_.expires_from_now() <= boost::posix_time::seconds(0))
    {
      // The timer has truly expired.
      socket_.close();

      // The timeout actor will enter an indefinite sleep, awaiting further
      // modifications to the expiry time by the connect or input actors.
      timeout_timer_.expires_at(boost::posix_time::pos_infin);
    }

    // Put the timeout actor back to sleep.
    timeout_timer_.async_wait(boost::bind(&client::handle_timeout, this));
  }

private:
  bool stopped_;
  tcp::socket socket_;
  asio::streambuf input_buffer_;
  asio::deadline_timer timeout_timer_;
  asio::deadline_timer heartbeat_timer_;
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
    tcp::resolver r(io_service);
    client c(io_service);

    c.start(r.resolve(tcp::resolver::query(argv[1], argv[2])));

    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
