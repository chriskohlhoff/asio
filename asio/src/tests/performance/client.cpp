//
// client.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "asio.hpp"
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/mem_fn.hpp>
#include <iostream>
#include <list>
#include <string>

using namespace asio;

class stats
{
public:
  stats()
    : mutex_(),
      total_bytes_written_(0),
      total_bytes_read_(0)
  {
  }

  void add(size_t bytes_written, size_t bytes_read)
  {
    detail::mutex::scoped_lock lock(mutex_);
    total_bytes_written_ += bytes_written;
    total_bytes_read_ += bytes_read;
  }

  void print()
  {
    detail::mutex::scoped_lock lock(mutex_);
    std::cout << total_bytes_written_ << " total bytes written\n";
    std::cout << total_bytes_read_ << " total bytes read\n";
  }

private:
  detail::mutex mutex_;
  size_t total_bytes_written_;
  size_t total_bytes_read_;
};

class session
{
public:
  session(io_service& ios, size_t block_size, stats& s)
    : strand_(ios),
      socket_(ios),
      block_size_(block_size),
      read_data_(new char[block_size]),
      read_data_length_(0),
      write_data_(new char[block_size]),
      unwritten_count_(0),
      bytes_written_(0),
      bytes_read_(0),
      stats_(s)
  {
    for (size_t i = 0; i < block_size_; ++i)
      write_data_[i] = static_cast<char>(i % 128);
  }

  ~session()
  {
    stats_.add(bytes_written_, bytes_read_);

    delete[] read_data_;
    delete[] write_data_;
  }

  void start(ip::tcp::resolver::iterator endpoint_iterator)
  {
    ip::tcp::endpoint endpoint = *endpoint_iterator;
    socket_.async_connect(endpoint,
        strand_.wrap(boost::bind(&session::handle_connect, this,
            placeholders::error, ++endpoint_iterator)));
  }

  void stop()
  {
    strand_.post(boost::bind(&session::close_socket, this));
  }

private:
  void handle_connect(const error& err,
      ip::tcp::resolver::iterator endpoint_iterator)
  {
    if (!err)
    {
      ++unwritten_count_;
      async_write(socket_, buffer(write_data_, block_size_),
          strand_.wrap(
            boost::bind(&session::handle_write, this, placeholders::error,
              placeholders::bytes_transferred)));
      socket_.async_read_some(buffer(read_data_, block_size_),
          strand_.wrap(
            boost::bind(&session::handle_read, this, placeholders::error,
              placeholders::bytes_transferred)));
    }
    else if (endpoint_iterator != ip::tcp::resolver::iterator())
    {
      socket_.close();
      ip::tcp::endpoint endpoint = *endpoint_iterator;
      socket_.async_connect(endpoint,
          strand_.wrap(boost::bind(&session::handle_connect, this,
              placeholders::error, ++endpoint_iterator)));
    }
  }

  void handle_read(const error& err, size_t length)
  {
    if (!err)
    {
      bytes_read_ += length;

      read_data_length_ = length;
      ++unwritten_count_;
      if (unwritten_count_ == 1)
      {
        std::swap(read_data_, write_data_);
        async_write(socket_, buffer(write_data_, read_data_length_),
            strand_.wrap(
              boost::bind(&session::handle_write, this, placeholders::error,
                placeholders::bytes_transferred)));
        socket_.async_read_some(buffer(read_data_, block_size_),
            strand_.wrap(
              boost::bind(&session::handle_read, this, placeholders::error,
                placeholders::bytes_transferred)));
      }
    }
  }

  void handle_write(const error& err, size_t length)
  {
    if (!err && length > 0)
    {
      bytes_written_ += length;

      --unwritten_count_;
      if (unwritten_count_ == 1)
      {
        std::swap(read_data_, write_data_);
        async_write(socket_, buffer(write_data_, read_data_length_),
            strand_.wrap(
              boost::bind(&session::handle_write, this, placeholders::error,
                placeholders::bytes_transferred)));
        socket_.async_read_some(buffer(read_data_, block_size_),
            strand_.wrap(
              boost::bind(&session::handle_read, this, placeholders::error,
                placeholders::bytes_transferred)));
      }
    }
  }

  void close_socket()
  {
    socket_.close();
  }

private:
  strand strand_;
  ip::tcp::socket socket_;
  size_t block_size_;
  char* read_data_;
  size_t read_data_length_;
  char* write_data_;
  int unwritten_count_;
  size_t bytes_written_;
  size_t bytes_read_;
  stats& stats_;
};

class client
{
public:
  client(io_service& ios, const ip::tcp::resolver::iterator endpoint_iterator,
      size_t block_size, size_t session_count, int timeout)
    : io_service_(ios),
      stop_timer_(ios),
      sessions_(),
      stats_()
  {
    stop_timer_.expires_from_now(boost::posix_time::seconds(timeout));
    stop_timer_.async_wait(boost::bind(&client::handle_timeout, this));

    for (size_t i = 0; i < session_count; ++i)
    {
      session* new_session = new session(io_service_, block_size, stats_);
      new_session->start(endpoint_iterator);
      sessions_.push_back(new_session);
    }
  }

  ~client()
  {
    while (!sessions_.empty())
    {
      delete sessions_.front();
      sessions_.pop_front();
    }

    stats_.print();
  }

  void handle_timeout()
  {
    std::for_each(sessions_.begin(), sessions_.end(),
        boost::mem_fn(&session::stop));
  }

private:
  io_service& io_service_;
  deadline_timer stop_timer_;
  std::list<session*> sessions_;
  stats stats_;
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 7)
    {
      std::cerr << "Usage: client <host> <port> <threads> <blocksize> ";
      std::cerr << "<sessions> <time>\n";
      return 1;
    }

    using namespace std; // For atoi.
    const char* host = argv[1];
    const char* port = argv[2];
    int thread_count = atoi(argv[3]);
    size_t block_size = atoi(argv[4]);
    size_t session_count = atoi(argv[5]);
    int timeout = atoi(argv[6]);

    io_service ios;

    ip::tcp::resolver r(ios);
    ip::tcp::resolver::iterator iter =
      r.resolve(ip::tcp::resolver::query(host, port));

    client c(ios, iter, block_size, session_count, timeout);

    std::list<thread*> threads;
    while (--thread_count > 0)
    {
      thread* new_thread = new thread(boost::bind(&io_service::run, &ios));
      threads.push_back(new_thread);
    }

    ios.run();

    while (!threads.empty())
    {
      threads.front()->join();
      delete threads.front();
      threads.pop_front();
    }
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
