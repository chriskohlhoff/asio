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
      std::cerr << "Usage: blocking_tcp_echo_client <host> <port>\n";
      return 1;
    }

    asio::demuxer d;

    asio::ipv4::host_resolver hr(d);
    asio::ipv4::host h;
    hr.get_host_by_name(argv[1], h);
    asio::ipv4::tcp::endpoint ep(atoi(argv[2]), h.addresses[0]);

    using namespace std; // For atoi and strlen.
    asio::stream_socket s(d);
    asio::socket_connector c(d);
    c.connect(s, ep);

    std::cout << "Enter message: ";
    char request[max_length];
    std::cin.getline(request, max_length);
    size_t request_length = strlen(request);
    asio::send_n(s, request, request_length);

    char reply[max_length];
    size_t reply_length = asio::recv_n(s, reply, request_length);
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
