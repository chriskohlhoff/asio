#include <iostream>
#include "asio.hpp"

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: client <host>" << std::endl;
      return 1;
    }

    asio::demuxer demuxer;

    asio::dgram_socket socket(demuxer, asio::ipv4::address(0));

    char send_buf[1] = { 0 };
    socket.sendto(send_buf, sizeof(send_buf), asio::ipv4::address(13, argv[1]));

    char recv_buf[128];
    asio::ipv4::address remote_address;
    size_t len = socket.recvfrom(recv_buf, sizeof(recv_buf), remote_address);
    std::cout.write(recv_buf, len);
  }
  catch (asio::socket_error& e)
  {
    std::cerr << e.what() << ": " << e.message() << std::endl;
  }

  return 0;
}
