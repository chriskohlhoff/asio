#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include "asio.hpp"

class session
{
public:
  session(asio::demuxer& d)
    : socket_(d)
  {
  }

  asio::stream_socket& socket()
  {
    return socket_;
  }

  void start()
  {
    socket_.async_recv(data_, max_length,
        boost::bind(&session::handle_recv, this, asio::arg::error,
          asio::arg::bytes_recvd));
  }

  void handle_recv(const asio::socket_error& error, size_t bytes_recvd)
  {
    if (!error && bytes_recvd > 0)
    {
      asio::async_send_n(socket_, data_, bytes_recvd,
          boost::bind(&session::handle_send, this, asio::arg::error,
            asio::arg::last_bytes_sent));
    }
    else
    {
      delete this;
    }
  }

  void handle_send(const asio::socket_error& error, size_t last_bytes_sent)
  {
    if (!error && last_bytes_sent > 0)
    {
      socket_.async_recv(data_, max_length,
          boost::bind(&session::handle_recv, this, asio::arg::error,
            asio::arg::bytes_recvd));
    }
    else
    {
      delete this;
    }
  }

private:
  asio::stream_socket socket_;
  enum { max_length = 1024 };
  char data_[max_length];
};

class server
{
public:
  server(asio::demuxer& d, short port)
    : demuxer_(d),
      acceptor_(d, asio::ipv4::address(port))
  {
    session* new_session = new session(demuxer_);
    acceptor_.async_accept(new_session->socket(),
        boost::bind(&server::handle_accept, this, new_session,
          asio::arg::error));
  }

  void handle_accept(session* new_session, const asio::socket_error& error)
  {
    if (!error)
    {
      new_session->start();
      new_session = new session(demuxer_);
      acceptor_.async_accept(new_session->socket(),
          boost::bind(&server::handle_accept, this, new_session,
            asio::arg::error));
    }
    else
    {
      delete new_session;
    }
  }

private:
  asio::demuxer& demuxer_;
  asio::socket_acceptor acceptor_;
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: async_tcp_echo_server <port>\n";
      return 1;
    }

    asio::demuxer d;

    using namespace std; // For atoi.
    server s(d, atoi(argv[1]));

    d.run();
  }
  catch (asio::socket_error& e)
  {
    std::cerr << "Socket error: " << e.message() << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
