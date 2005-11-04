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
      std::cerr << "Usage: blocking_tcp_echo_client <host> <port>\n";
      return 1;
    }

    asio::demuxer d;

    using namespace std; // For atoi and strlen.
    asio::ipv4::host_resolver hr(d);
    asio::ipv4::host h;
    hr.get_host_by_name(h, argv[1]);
    asio::ipv4::tcp::endpoint ep(atoi(argv[2]), h.address(0));

    asio::stream_socket s(d);
    s.connect(ep);

    std::cout << "Enter message: ";
    char request[max_length];
    std::cin.getline(request, max_length);
    size_t request_length = strlen(request);
    asio::write_n(s, asio::buffer(request, request_length));

    char reply[max_length];
    size_t reply_length = asio::read_n(s,
        asio::buffer(reply, request_length));
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
