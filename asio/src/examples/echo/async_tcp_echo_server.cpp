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
    socket_.async_read(data_, max_length,
        boost::bind(&session::handle_read, this, asio::placeholders::error,
          asio::placeholders::bytes_transferred));
  }

  void handle_read(const asio::error& error, size_t bytes_transferred)
  {
    if (!error && bytes_transferred > 0)
    {
      asio::async_write_n(socket_, data_, bytes_transferred,
          boost::bind(&session::handle_write, this, asio::placeholders::error,
            asio::placeholders::bytes_transferred));
    }
    else
    {
      delete this;
    }
  }

  void handle_write(const asio::error& error, size_t last_bytes_transferred)
  {
    if (!error && last_bytes_transferred > 0)
    {
      socket_.async_read(data_, max_length,
          boost::bind(&session::handle_read, this, asio::placeholders::error,
            asio::placeholders::bytes_transferred));
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
      acceptor_(d, asio::ipv4::tcp::endpoint(port))
  {
    session* new_session = new session(demuxer_);
    acceptor_.async_accept(new_session->socket(),
        boost::bind(&server::handle_accept, this, new_session,
          asio::placeholders::error));
  }

  void handle_accept(session* new_session, const asio::error& error)
  {
    if (!error)
    {
      new_session->start();
      new_session = new session(demuxer_);
      acceptor_.async_accept(new_session->socket(),
          boost::bind(&server::handle_accept, this, new_session,
            asio::placeholders::error));
    }
    else if (error == asio::error::connection_aborted)
    {
      acceptor_.async_accept(new_session->socket(),
          boost::bind(&server::handle_accept, this, new_session,
            asio::placeholders::error));
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
  catch (asio::error& e)
  {
    std::cerr << e << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
