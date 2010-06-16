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

using namespace asio;
using asio::ip::tcp;

//
// This class consists of three asynchronous "actors":
//
//        Protocol Actor                      Timeout Actor
//        ~~~~~~~~~~~~~~                      ~~~~~~~~~~~~~
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
//                      |
//         +------------+------+            Heartbeat Actor
//         |                   |            ~~~~~~~~~~~~~~~
//         V                   |
//  +------------+             |     +-------------+
//  |            |             |     |             |
//  | start_read |<----+       +---->| start_write |<----+
//  |            |     |             |             |     |
//  +------------+     |             +-------------+     |
//         |           |                    |            |
//         |    +-------------+             |     +--------------+
//         |    |             |             |     |              |
//         +--->| handle_read |             +---->| handle_write |
//              |             |                   |              |
//              +-------------+                   +--------------+
//
// The protocol actor manages all socket operations, such as connection
// establishment.
//
// The timeout actor is responsible for managing timeouts. When a timeout
// occurs it will close the socket. This will cause any pending asynchronous
// operations to complete with the operation_aborted error.
//
class client
{
public:
  client(io_service& i)
    : stopped_(false),
      socket_(i),
      input_timer_(i),
      heartbeat_timer_(i)
  {
  }

  // Called by the user of the client class to initiate the connection process.
  // The endpoint iterator will have been obtained using a tcp::resolver.
  void start(tcp::resolver::iterator endpoint_iter)
  {
    // Start the protocol actor.
    start_connect(endpoint_iter);

    // Start the timeout actor. You will note that we're not setting any
    // particular timeout here. Instead, the protocol actor will update the
    // timeout prior to each asynchronous operation.
    input_timer_.async_wait(boost::bind(&client::handle_timeout, this));
  }

  // This function terminates the two actors to shut down the connection. It
  // may be called by the user of the client class, or by the class itself in
  // response to graceful termination or an unrecoverable error.
  void stop()
  {
    stopped_ = true;
    socket_.close();
    input_timer_.cancel();
    heartbeat_timer_.cancel();
  }

private:
  void start_connect(tcp::resolver::iterator endpoint_iter)
  {
    if (endpoint_iter != tcp::resolver::iterator())
    {
      std::cout << "Trying " << endpoint_iter->endpoint() << "...\n";

      // Set a timeout for the connect operation.
      input_timer_.expires_from_now(boost::posix_time::seconds(2));

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

      // Start receiving messages from the server.
      start_read();

      // Start the heartbeat actor.
      start_write();
    }
  }

  void start_read()
  {
    // Set a timeout for the read operation.
    input_timer_.expires_from_now(boost::posix_time::seconds(30));

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
    if (input_timer_.expires_from_now() <= boost::posix_time::seconds(0))
    {
      // The timer has truly expired.
      socket_.close();

      // The timeout actor will enter an indefinite sleep, awaiting further
      // modifications to the expiry time by the protocol actor.
      input_timer_.expires_at(boost::posix_time::pos_infin);
    }

    // Put the timeout actor back to sleep.
    input_timer_.async_wait(boost::bind(&client::handle_timeout, this));
  }

private:
  bool stopped_;
  tcp::socket socket_;
  asio::streambuf input_buffer_;
  deadline_timer input_timer_;
  deadline_timer heartbeat_timer_;
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

    io_service i;
    tcp::resolver r(i);
    client c(i);

    c.start(r.resolve(tcp::resolver::query(argv[1], argv[2])));

    i.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
