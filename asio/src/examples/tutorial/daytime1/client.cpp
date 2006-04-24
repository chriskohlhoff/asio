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

    asio::ip::tcp::resolver resolver(io_service);
    asio::ip::tcp::resolver::query query(argv[1], "daytime");
    asio::ip::tcp::resolver::iterator endpoint_iterator =
      resolver.resolve(query);
    asio::ip::tcp::resolver::iterator end;

    asio::ip::tcp::socket socket(io_service);
    asio::error error = asio::error::host_not_found;
    while (error && endpoint_iterator != end)
    {
      socket.close();
      socket.connect(*endpoint_iterator++, asio::assign_error(error));
    }
    if (error)
      throw error;

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
