//
// client.hpp
// ~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
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
      total_bytes_sent_(0),
      total_bytes_recvd_(0)
  {
  }

  void add(size_t bytes_sent, size_t bytes_recvd)
  {
    detail::mutex::scoped_lock lock(mutex_);
    total_bytes_sent_ += bytes_sent;
    total_bytes_recvd_ += bytes_recvd;
  }

  void print()
  {
    detail::mutex::scoped_lock lock(mutex_);
    std::cout << total_bytes_sent_ << " total bytes sent\n";
    std::cout << total_bytes_recvd_ << " total bytes received\n";
  }

private:
  detail::mutex mutex_;
  size_t total_bytes_sent_;
  size_t total_bytes_recvd_;
};

class session
{
public:
  session(demuxer& d, size_t block_size, stats& s)
    : dispatcher_(d),
      socket_(d),
      block_size_(block_size),
      recv_data_(new char[block_size]),
      recv_data_length_(0),
      send_data_(new char[block_size]),
      unsent_count_(0),
      bytes_sent_(0),
      bytes_recvd_(0),
      stats_(s)
  {
    for (size_t i = 0; i < block_size_; ++i)
      send_data_[i] = static_cast<char>(i % 128);
  }

  ~session()
  {
    stats_.add(bytes_sent_, bytes_recvd_);

    delete[] recv_data_;
    delete[] send_data_;
  }

  stream_socket& socket()
  {
    return socket_;
  }

  void start()
  {
    ++unsent_count_;
    async_send_n(socket_, send_data_, block_size_, dispatcher_.wrap(
          boost::bind(&session::handle_send, this, arg::error,
            arg::last_bytes_sent, arg::total_bytes_sent)));
    socket_.async_recv(recv_data_, block_size_, dispatcher_.wrap(
          boost::bind(&session::handle_recv, this, arg::error,
            arg::bytes_recvd)));
  }

  void stop()
  {
    dispatcher_.post(boost::bind(&stream_socket::close, &socket_));
  }

  void handle_recv(const error& err, size_t length)
  {
    if (!err && length > 0)
    {
      bytes_recvd_ += length;

      recv_data_length_ = length;
      ++unsent_count_;
      if (unsent_count_ == 1)
      {
        std::swap(recv_data_, send_data_);
        async_send_n(socket_, send_data_, recv_data_length_, dispatcher_.wrap(
              boost::bind(&session::handle_send, this, arg::error,
                arg::last_bytes_sent, arg::total_bytes_sent)));
        socket_.async_recv(recv_data_, block_size_, dispatcher_.wrap(
              boost::bind(&session::handle_recv, this, arg::error,
                arg::bytes_recvd)));
      }
    }
  }

  void handle_send(const error& err, size_t last_length, size_t total_length)
  {
    if (!err && last_length > 0)
    {
      bytes_sent_ += total_length;

      --unsent_count_;
      if (unsent_count_ == 1)
      {
        std::swap(recv_data_, send_data_);
        async_send_n(socket_, send_data_, recv_data_length_, dispatcher_.wrap(
              boost::bind(&session::handle_send, this, arg::error,
                arg::last_bytes_sent, arg::total_bytes_sent)));
        socket_.async_recv(recv_data_, block_size_, dispatcher_.wrap(
              boost::bind(&session::handle_recv, this, arg::error,
                arg::bytes_recvd)));
      }
    }
  }

private:
  locking_dispatcher dispatcher_;
  stream_socket socket_;
  size_t block_size_;
  char* recv_data_;
  size_t recv_data_length_;
  char* send_data_;
  int unsent_count_;
  size_t bytes_sent_;
  size_t bytes_recvd_;
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
            new_session, arg::error)));

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
                new_session, arg::error)));
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
