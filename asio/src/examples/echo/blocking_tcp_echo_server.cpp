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
    char data[max_length];

    int length;
    while ((length = sock->recv(data, max_length)) > 0)
      if (asio::send_n(*sock, data, length) <= 0)
        break;
  }
  catch (asio::socket_error& e)
  {
    std::cerr << "Socket error in thread: " << e.message() << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception in thread: " << e.what() << "\n";
  }
}

void server(asio::demuxer& d, short port)
{
  asio::socket_acceptor a(d, asio::inet_address_v4(port));
  for (;;)
  {
    stream_socket_ptr sock(new asio::stream_socket(d));
    a.accept(*sock);
    asio::detail::thread t(boost::bind(session, sock));
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

    asio::demuxer d;

    using namespace std; // For atoi.
    server(d, atoi(argv[1]));
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
