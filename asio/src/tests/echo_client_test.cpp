#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>
#include <string.h>

using namespace asio;

class echo_session
{
public:
  echo_session(demuxer& d)
    : socket_(d),
      msg_count_(0)
  {
  }

  stream_socket& socket()
  {
    return socket_;
  }

  void start()
  {
    for (int i = 0; i < max_length; ++i)
      data_[i] = i % 128;

    socket_.async_send_n(data_, max_length,
        boost::bind(&echo_session::handle_send, this, _1, _2, _3));
  }

  void handle_recv(const socket_error& error, size_t length)
  {
    if (!error && length > 0 && ++msg_count_ < 10000)
    {
      socket_.async_send_n(data_, length,
          boost::bind(&echo_session::handle_send, this, _1, _2, _3));
    }
    else
    {
      delete this;
    }
  }

  void handle_send(const socket_error& error, size_t length, size_t last_length)
  {
    if (!error && last_length > 0)
    {
      socket_.async_recv(data_, max_length,
          boost::bind(&echo_session::handle_recv, this, _1, _2));
    }
    else
    {
      delete this;
    }
  }

private:
  stream_socket socket_;
  enum { max_length = 512 };
  char data_[max_length];
  int msg_count_;
};

class echo_client
{
public:
  echo_client(demuxer& d)
    : demuxer_(d),
      connector_(d),
      connection_count_(0)
  {
    echo_session* new_session = new echo_session(demuxer_);
    connector_.async_connect(new_session->socket(),
        inet_address_v4(12345, "localhost"),
        boost::bind(&echo_client::handle_connect, this, new_session, _1));
  }

  void handle_connect(echo_session* session, const socket_error& error)
  {
    if (!error)
    {
      if (++connection_count_ < 10)
      {
        echo_session* new_session = new echo_session(demuxer_);
        connector_.async_connect(new_session->socket(),
            inet_address_v4(12345, "localhost"),
            boost::bind(&echo_client::handle_connect, this, new_session, _1));
      }

      session->start();
    }
    else
    {
      delete session;
    }
  }

private:
  demuxer& demuxer_;
  socket_connector connector_;
  int connection_count_;
};

int main()
{
  try
  {
    demuxer d;
    echo_client e(d);
    d.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
