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

    asio::io_service io_service;

    asio::ipv4::udp::socket s(io_service,
        asio::ipv4::udp::endpoint(0));

    asio::ipv4::host_resolver hr(io_service);
    asio::ipv4::host h;
    hr.get_host_by_name(h, argv[1]);
    asio::ipv4::udp::endpoint receiver_endpoint(
        atoi(argv[2]), h.address(0));

    using namespace std; // For atoi and strlen.
    std::cout << "Enter message: ";
    char request[max_length];
    std::cin.getline(request, max_length);
    size_t request_length = strlen(request);
    s.connect(receiver_endpoint);
    s.send(asio::buffer(request, request_length), 0);

    char reply[max_length];
    asio::ipv4::udp::endpoint sender_endpoint;
    size_t reply_length = s.receive_from(
        asio::buffer(reply, max_length), 0, sender_endpoint);
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
