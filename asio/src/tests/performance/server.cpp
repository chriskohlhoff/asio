//
// server.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "asio.hpp"
#include <algorithm>
#include <boost/bind.hpp>
#include <iostream>
#include <list>

using namespace asio;

class session
{
public:
  session(demuxer& d, size_t block_size)
    : demuxer_(d),
      dispatcher_(d),
      socket_(d),
      block_size_(block_size),
      read_data_(new char[block_size]),
      read_data_length_(0),
      write_data_(new char[block_size]),
      unsent_count_(0),
      op_count_(0)
  {
  }

  ~session()
  {
    delete[] read_data_;
    delete[] write_data_;
  }

  stream_socket& socket()
  {
    return socket_;
  }

  void start()
  {
    ++op_count_;
    socket_.async_read(read_data_, block_size_,
        dispatcher_.wrap(
          boost::bind(&session::handle_read, this, placeholders::error,
            placeholders::bytes_transferred)));
  }

  void handle_read(const error& err, size_t length)
  {
    --op_count_;

    if (!err && length > 0)
    {
      read_data_length_ = length;
      ++unsent_count_;
      if (unsent_count_ == 1)
      {
        op_count_ += 2;
        std::swap(read_data_, write_data_);
        async_write_n(socket_, write_data_, read_data_length_,
            dispatcher_.wrap(
              boost::bind(&session::handle_write, this, placeholders::error,
                placeholders::last_bytes_transferred)));
        socket_.async_read(read_data_, block_size_,
            dispatcher_.wrap(
              boost::bind(&session::handle_read, this, placeholders::error,
                placeholders::bytes_transferred)));
      }
    }

    if (op_count_ == 0)
      demuxer_.post(boost::bind(&session::destroy, this));
  }

  void handle_write(const error& err, size_t last_length)
  {
    --op_count_;

    if (!err && last_length > 0)
    {
      --unsent_count_;
      if (unsent_count_ == 1)
      {
        op_count_ += 2;
        std::swap(read_data_, write_data_);
        async_write_n(socket_, write_data_, read_data_length_,
            dispatcher_.wrap(
              boost::bind(&session::handle_write, this, placeholders::error,
                placeholders::last_bytes_transferred)));
        socket_.async_read(read_data_, block_size_,
            dispatcher_.wrap(
              boost::bind(&session::handle_read, this, placeholders::error,
                placeholders::bytes_transferred)));
      }
    }

    if (op_count_ == 0)
      demuxer_.post(boost::bind(&session::destroy, this));
  }

  static void destroy(session* s)
  {
    delete s;
  }

private:
  demuxer& demuxer_;
  locking_dispatcher dispatcher_;
  stream_socket socket_;
  size_t block_size_;
  char* read_data_;
  size_t read_data_length_;
  char* write_data_;
  int unsent_count_;
  int op_count_;
};

class server
{
public:
  server(demuxer& d, const ipv4::tcp::endpoint& endpoint, size_t block_size)
    : demuxer_(d),
      acceptor_(d),
      block_size_(block_size)
  {
    acceptor_.open(ipv4::tcp());
    acceptor_.set_option(socket_option::reuse_address(1));
    acceptor_.bind(endpoint);
    acceptor_.listen();

    session* new_session = new session(demuxer_, block_size_);
    acceptor_.async_accept(new_session->socket(),
        boost::bind(&server::handle_accept, this, new_session,
          placeholders::error));
  }

  void handle_accept(session* new_session, const error& err)
  {
    if (!err)
    {
      new_session->start();
      new_session = new session(demuxer_, block_size_);
      acceptor_.async_accept(new_session->socket(),
          boost::bind(&server::handle_accept, this, new_session,
            placeholders::error));
    }
    else
    {
      delete new_session;
    }
  }

private:
  demuxer& demuxer_;
  socket_acceptor acceptor_;
  size_t block_size_;
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 4)
    {
      std::cerr << "Usage: server <port> <threads> <blocksize>\n";
      return 1;
    }

    using namespace std; // For atoi.
    short port = atoi(argv[1]);
    int thread_count = atoi(argv[2]);
    size_t block_size = atoi(argv[3]);

    demuxer d;

    server s(d, ipv4::tcp::endpoint(port), block_size);

    // Threads not currently supported in this test.
    std::list<thread*> threads;
    while (--thread_count > 0)
    {
      thread* new_thread = new thread(boost::bind(&demuxer::run, &d));
      threads.push_back(new_thread);
    }

    d.run();

    while (!threads.empty())
    {
      threads.front()->join();
      delete threads.front();
      threads.pop_front();
    }
  }
  catch (asio::error& e)
  {
    std::cerr << "Exception: " << e << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
