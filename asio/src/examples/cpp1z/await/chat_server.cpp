//
// chat_server.cpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <cstdlib>
#include <deque>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <asio/await.hpp>
#include <asio/detached.hpp>
#include <asio/redirect_error.hpp>
#include <asio/signal_set.hpp>
#include <asio/ts/networking.hpp>

using asio::ip::tcp;

//----------------------------------------------------------------------

class chat_participant
{
public:
  virtual ~chat_participant() {}
  virtual void deliver(const std::string& msg) = 0;
};

typedef std::shared_ptr<chat_participant> chat_participant_ptr;

//----------------------------------------------------------------------

class chat_room
{
public:
  void join(chat_participant_ptr participant)
  {
    participants_.insert(participant);
    for (auto msg: recent_msgs_)
      participant->deliver(msg);
  }

  void leave(chat_participant_ptr participant)
  {
    participants_.erase(participant);
  }

  void deliver(const std::string& msg)
  {
    recent_msgs_.push_back(msg);
    while (recent_msgs_.size() > max_recent_msgs)
      recent_msgs_.pop_front();

    for (auto participant: participants_)
      participant->deliver(msg);
  }

private:
  std::set<chat_participant_ptr> participants_;
  enum { max_recent_msgs = 100 };
  std::deque<std::string> recent_msgs_;
};

//----------------------------------------------------------------------

class chat_session
  : public chat_participant,
    public std::enable_shared_from_this<chat_session>
{
public:
  chat_session(tcp::socket socket, chat_room& room)
    : socket_(std::move(socket)),
      timer_(socket_.get_executor().context()),
      room_(room)
  {
    timer_.expires_at(std::chrono::steady_clock::time_point::max());
  }

  void start()
  {
    room_.join(shared_from_this());

    asio::spawn(socket_.get_executor(),
        &chat_session::reader, shared_from_this(), asio::detached);

    asio::spawn(socket_.get_executor(),
        &chat_session::writer, shared_from_this(), asio::detached);
  }

  void deliver(const std::string& msg)
  {
    write_msgs_.push_back(msg);
    timer_.cancel_one();
  }

private:
  asio::awaitable<void> reader(asio::await_context ctx)
  {
    try
    {
      for (std::string read_msg;;)
      {
        std::size_t n = co_await asio::async_read_until(socket_,
            asio::dynamic_buffer(read_msg, 1024), "\n", ctx);

        room_.deliver(read_msg.substr(0, n));
        read_msg.erase(0, n);
      }
    }
    catch (std::exception&)
    {
      stop();
    }
  }

  asio::awaitable<void> writer(asio::await_context ctx)
  {
    try
    {
      while (socket_.is_open())
      {
        if (write_msgs_.empty())
        {
          asio::error_code ec;
          co_await timer_.async_wait(asio::redirect_error(ctx, ec));
        }
        else
        {
          co_await asio::async_write(socket_,
              asio::buffer(write_msgs_.front()), ctx);
          write_msgs_.pop_front();
        }
      }
    }
    catch (std::exception&)
    {
      stop();
    }
  }

  void stop()
  {
    room_.leave(shared_from_this());
    socket_.close();
    timer_.cancel();
  }

  tcp::socket socket_;
  asio::steady_timer timer_;
  chat_room& room_;
  std::deque<std::string> write_msgs_;
};

//----------------------------------------------------------------------

asio::awaitable<void> listener(tcp::acceptor acceptor, asio::await_context ctx)
{
  chat_room room;

  for (;;)
  {
    std::make_shared<chat_session>(
        co_await acceptor.async_accept(ctx),
        room
      )->start();
  }
}

//----------------------------------------------------------------------

int main(int argc, char* argv[])
{
  try
  {
    if (argc < 2)
    {
      std::cerr << "Usage: chat_server <port> [<port> ...]\n";
      return 1;
    }

    asio::io_context io_context;

    for (int i = 1; i < argc; ++i)
    {
      unsigned short port = std::atoi(argv[i]);
      asio::spawn(io_context, listener,
          tcp::acceptor(io_context, {tcp::v4(), port}),
          asio::detached);
    }

    asio::signal_set signals(io_context, SIGINT, SIGTERM);
    signals.async_wait([&](auto, auto){ io_context.stop(); });

    io_context.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
