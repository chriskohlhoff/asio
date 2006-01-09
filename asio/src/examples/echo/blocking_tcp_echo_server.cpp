#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/smart_ptr.hpp>
#include "asio.hpp"

const int max_length = 1024;

typedef boost::shared_ptr<asio::stream_socket> stream_socket_ptr;

void session(stream_socket_ptr sock)
{
  try
  {
    for (;;)
    {
      char data[max_length];

      asio::error error;
      size_t length = sock->read_some(
          asio::buffer(data), asio::assign_error(error));
      if (error == asio::error::eof)
        break; // Connection closed cleanly by peer.
      else if (error)
        throw error; // Some other error.

      asio::write(*sock, asio::buffer(data, length));
    }
  }
  catch (asio::error& e)
  {
    std::cerr << "Error in thread: " << e << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception in thread: " << e.what() << "\n";
  }
}

void server(asio::io_service& io_service, short port)
{
  asio::socket_acceptor a(io_service,
      asio::ipv4::tcp::endpoint(port));
  for (;;)
  {
    stream_socket_ptr sock(new asio::stream_socket(io_service));
    asio::error error;
    a.accept(*sock, asio::assign_error(error));
    if (!error)
      asio::thread t(boost::bind(session, sock));
    else if (error != asio::error::connection_aborted)
      throw error;
  }
}

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: blocking_tcp_echo_server <port>\n";
      return 1;
    }

    asio::io_service io_service;

    using namespace std; // For atoi.
    server(io_service, atoi(argv[1]));
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
