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

    asio::stream_socket socket(demuxer);

    asio::socket_connector connector(demuxer);
    connector.connect(socket, asio::ipv4::address(13, argv[1]));

    char buf[128];
    while (size_t len = socket.recv(buf, sizeof(buf)))
      std::cout.write(buf, len);
  }
  catch (asio::socket_error& e)
  {
    std::cerr << e.what() << ": " << e.message() << std::endl;
  }

  return 0;
}
