#include <cstdlib>
#include <iostream>
#include "asio.hpp"

enum { max_length = 1024 };

void server(asio::io_service& io_service, short port)
{
  asio::datagram_socket sock(io_service,
      asio::ipv4::udp::endpoint(port));
  for (;;)
  {
    char data[max_length];
    asio::ipv4::udp::endpoint sender_endpoint;
    size_t length = sock.receive_from(
        asio::buffer(data, max_length), 0, sender_endpoint);
    sock.send_to(asio::buffer(data, length), 0, sender_endpoint);
  }
}

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: blocking_udp_echo_server <port>\n";
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
