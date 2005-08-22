//
// client.hpp
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
  session(demuxer& d, size_t block_size, stats& s)
    : dispatcher_(d),
      socket_(d),
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

  stream_socket& socket()
  {
    return socket_;
  }

  void start()
  {
    ++unwritten_count_;
    async_write_n(socket_, write_data_, block_size_,
        dispatcher_.wrap(
          boost::bind(&session::handle_write, this, placeholders::error,
            placeholders::last_bytes_transferred,
            placeholders::total_bytes_transferred)));
    socket_.async_read(read_data_, block_size_,
        dispatcher_.wrap(
          boost::bind(&session::handle_read, this, placeholders::error,
            placeholders::bytes_transferred)));
  }

  void stop()
  {
    dispatcher_.post(boost::bind(&stream_socket::close, &socket_));
  }

  void handle_read(const error& err, size_t length)
  {
    if (!err && length > 0)
    {
      bytes_read_ += length;

      read_data_length_ = length;
      ++unwritten_count_;
      if (unwritten_count_ == 1)
      {
        std::swap(read_data_, write_data_);
        async_write_n(socket_, write_data_, read_data_length_,
            dispatcher_.wrap(
              boost::bind(&session::handle_write, this, placeholders::error,
                placeholders::last_bytes_transferred,
                placeholders::total_bytes_transferred)));
        socket_.async_read(read_data_, block_size_,
            dispatcher_.wrap(
              boost::bind(&session::handle_read, this, placeholders::error,
                placeholders::bytes_transferred)));
      }
    }
  }

  void handle_write(const error& err, size_t last_length, size_t total_length)
  {
    if (!err && last_length > 0)
    {
      bytes_written_ += total_length;

      --unwritten_count_;
      if (unwritten_count_ == 1)
      {
        std::swap(read_data_, write_data_);
        async_write_n(socket_, write_data_, read_data_length_,
            dispatcher_.wrap(
              boost::bind(&session::handle_write, this, placeholders::error,
                placeholders::last_bytes_transferred,
                placeholders::total_bytes_transferred)));
        socket_.async_read(read_data_, block_size_,
            dispatcher_.wrap(
              boost::bind(&session::handle_read, this, placeholders::error,
                placeholders::bytes_transferred)));
      }
    }
  }

private:
  locking_dispatcher dispatcher_;
  stream_socket socket_;
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
  client(demuxer& d, const ipv4::tcp::endpoint& server_endpoint,
      size_t block_size, size_t session_count, int timeout)
    : demuxer_(d),
      dispatcher_(d),
      stop_timer_(d, asio::time::now() + timeout),
      connector_(d),
      server_endpoint_(server_endpoint),
      block_size_(block_size),
      max_session_count_(session_count),
      sessions_(),
      stats_()
  {
    session* new_session = new session(demuxer_, block_size, stats_);
    connector_.async_connect(new_session->socket(), server_endpoint_,
        dispatcher_.wrap(boost::bind(&client::handle_connect, this,
            new_session, placeholders::error)));

    stop_timer_.async_wait(dispatcher_.wrap(
          boost::bind(&client::handle_timeout, this)));
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

  void handle_connect(session* new_session, const error& err)
  {
    if (!err)
    {
      sessions_.push_back(new_session);
      new_session->start();

      if (sessions_.size() < max_session_count_)
      {
        new_session = new session(demuxer_, block_size_, stats_);
        connector_.async_connect(new_session->socket(), server_endpoint_,
            dispatcher_.wrap(boost::bind(&client::handle_connect, this,
                new_session, placeholders::error)));
      }
    }
    else
    {
      delete new_session;
    }
  }

private:
  demuxer& demuxer_;
  locking_dispatcher dispatcher_;
  timer stop_timer_;
  socket_connector connector_;
  ipv4::tcp::endpoint server_endpoint_;
  size_t block_size_;
  size_t max_session_count_;
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
    short port = atoi(argv[2]);
    int thread_count = atoi(argv[3]);
    size_t block_size = atoi(argv[4]);
    size_t session_count = atoi(argv[5]);
    int timeout = atoi(argv[6]);

    demuxer d;

    ipv4::host_resolver hr(d);
    ipv4::host h;
    hr.get_host_by_name(h, host);
    ipv4::tcp::endpoint ep(port, h.addresses[0]);

    client c(d, ep, block_size, session_count, timeout);

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
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
