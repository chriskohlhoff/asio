#include <cstdlib>
#include <iostream>
#include "asio.hpp"

const int max_length = 1024;

void server(asio::demuxer& d, short port)
{
  asio::dgram_socket sock(d, asio::ipv4::address(port));
  for (;;)
  {
    char data[max_length];
    asio::ipv4::address sender_address;
    size_t length = sock.recvfrom(data, max_length, sender_address);
    sock.sendto(data, length, sender_address);
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
