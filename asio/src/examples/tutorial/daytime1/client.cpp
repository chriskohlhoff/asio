#include <iostream>
#include <asio.hpp>

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: client <host>" << std::endl;
      return 1;
    }

    asio::io_service io_service;

    asio::ipv4::host_resolver host_resolver(io_service);
    asio::ipv4::host host;
    host_resolver.get_host_by_name(host, argv[1]);
    asio::ipv4::tcp::endpoint remote_endpoint(13, host.address(0));

    asio::ipv4::tcp::socket socket(io_service);
    socket.connect(remote_endpoint);

    for (;;)
    {
      char buf[128];
      asio::error error;
      size_t len = socket.read_some(
          asio::buffer(buf), asio::assign_error(error));
      if (error == asio::error::eof)
        break; // Connection closed cleanly by peer.
      else if (error)
        throw error; // Some other error.
      std::cout.write(buf, len);
    }
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
