#include <cstdlib>
#include <cstring>
#include <iostream>
#include "asio.hpp"

const int max_length = 1024;

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 3)
    {
      std::cerr << "Usage: blocking_udp_echo_client <host> <port>\n";
      return 1;
    }

    asio::demuxer d;

    asio::dgram_socket s(d, asio::ipv4::address(0));

    using namespace std; // For atoi and strlen.
    std::cout << "Enter message: ";
    char request[max_length];
    std::cin.getline(request, max_length);
    size_t request_length = strlen(request);
    asio::ipv4::address receiver_address(atoi(argv[2]), argv[1]);
    s.sendto(request, request_length, receiver_address);

    char reply[max_length];
    asio::ipv4::address sender_address;
    size_t reply_length = s.recvfrom(reply, max_length, sender_address);
    std::cout << "Reply is: ";
    std::cout.write(reply, reply_length);
    std::cout << "\n";
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
