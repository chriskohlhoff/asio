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

    asio::ipv4::host_resolver host_resolver(demuxer);
    asio::ipv4::host host;
    host_resolver.get_host_by_name(argv[1], host);
    asio::ipv4::tcp::endpoint remote_endpoint(13, host.addresses[0]);

    asio::stream_socket socket(demuxer);

    asio::socket_connector connector(demuxer);
    connector.connect(socket, remote_endpoint);

    char buf[128];
    while (size_t len = socket.recv(buf, sizeof(buf)))
      std::cout.write(buf, len);
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
