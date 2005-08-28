#include <cstdlib>
#include <cstring>
#include <iostream>
#include "asio.hpp"

enum { max_length = 1024 };

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

    asio::datagram_socket s(d, asio::ipv4::udp::endpoint(0));

    asio::ipv4::host_resolver hr(d);
    asio::ipv4::host h;
    hr.get_host_by_name(h, argv[1]);
    asio::ipv4::udp::endpoint receiver_endpoint(atoi(argv[2]), h.addresses[0]);

    using namespace std; // For atoi and strlen.
    std::cout << "Enter message: ";
    char request[max_length];
    std::cin.getline(request, max_length);
    size_t request_length = strlen(request);
    s.send_to(request, request_length, 0, receiver_endpoint);

    char reply[max_length];
    asio::ipv4::udp::endpoint sender_endpoint;
    size_t reply_length = s.receive_from(reply, max_length, 0, sender_endpoint);
    std::cout << "Reply is: ";
    std::cout.write(reply, reply_length);
    std::cout << "\n";
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
