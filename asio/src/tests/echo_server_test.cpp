#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

class echo_session
{
public:
  echo_session(demuxer& d)
    : socket_(d)
  {
  }

  stream_socket& socket()
  {
    return socket_;
  }

  void start()
  {
    socket_.async_recv(data_, max_length,
        boost::bind(&echo_session::handle_recv, this, _1, _2));
  }

  void handle_recv(const socket_error& error, size_t length)
  {
    if (!error && length > 0)
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
};

class echo_server
{
public:
  echo_server(demuxer& d)
    : demuxer_(d),
      acceptor_(d, inet_address_v4(12345))
  {
    echo_session* new_session = new echo_session(demuxer_);
    acceptor_.async_accept(new_session->socket(),
        boost::bind(&echo_server::handle_accept, this, new_session, _1));
  }

  void handle_accept(echo_session* session, const socket_error& error)
  {
    if (!error)
    {
      echo_session* new_session = new echo_session(demuxer_);
      acceptor_.async_accept(new_session->socket(),
          boost::bind(&echo_server::handle_accept, this, new_session, _1));

      session->start();
    }
    else
    {
      delete session;
    }
  }

private:
  demuxer& demuxer_;
  socket_acceptor acceptor_;
};

int main()
{
  try
  {
    demuxer d;
    echo_server e(d);
    d.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
