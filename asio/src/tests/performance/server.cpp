//
// server.hpp
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
      recv_data_(new char[block_size]),
      recv_data_length_(0),
      send_data_(new char[block_size]),
      unsent_count_(0),
      op_count_(0)
  {
  }

  ~session()
  {
    delete[] recv_data_;
    delete[] send_data_;
  }

  stream_socket& socket()
  {
    return socket_;
  }

  void start()
  {
    ++op_count_;
    socket_.async_recv(recv_data_, block_size_,
        dispatcher_.wrap(boost::bind(&session::handle_recv, this, arg::error,
            arg::bytes_recvd)));
  }

  void handle_recv(const socket_error& error, size_t length)
  {
    --op_count_;

    if (!error && length > 0)
    {
      recv_data_length_ = length;
      ++unsent_count_;
      if (unsent_count_ == 1)
      {
        op_count_ += 2;
        std::swap(recv_data_, send_data_);
        async_send_n(socket_, send_data_, recv_data_length_, dispatcher_.wrap(
              boost::bind(&session::handle_send, this, arg::error,
                arg::last_bytes_sent)));
        socket_.async_recv(recv_data_, block_size_, dispatcher_.wrap(
              boost::bind(&session::handle_recv, this, arg::error,
                arg::bytes_recvd)));
      }
    }

    if (op_count_ == 0)
      demuxer_.post(boost::bind(&session::destroy, this));
  }

  void handle_send(const socket_error& error, size_t last_length)
  {
    --op_count_;

    if (!error && last_length > 0)
    {
      --unsent_count_;
      if (unsent_count_ == 1)
      {
        op_count_ += 2;
        std::swap(recv_data_, send_data_);
        async_send_n(socket_, send_data_, recv_data_length_, dispatcher_.wrap(
              boost::bind(&session::handle_send, this, arg::error,
                arg::last_bytes_sent)));
        socket_.async_recv(recv_data_, block_size_, dispatcher_.wrap(
              boost::bind(&session::handle_recv, this, arg::error,
                arg::bytes_recvd)));
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
  char* recv_data_;
  size_t recv_data_length_;
  char* send_data_;
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
        boost::bind(&server::handle_accept, this, new_session, arg::error));
  }

  void handle_accept(session* new_session, const socket_error& error)
  {
    if (!error)
    {
      new_session->start();
      new_session = new session(demuxer_, block_size_);
      acceptor_.async_accept(new_session->socket(),
          boost::bind(&server::handle_accept, this, new_session, arg::error));
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
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
