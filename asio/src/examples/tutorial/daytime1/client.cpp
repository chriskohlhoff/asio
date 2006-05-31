#include <iostream>
#include <boost/array.hpp>
#include <asio.hpp>

using asio::ip::tcp;

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

    tcp::resolver resolver(io_service);
    tcp::resolver::query query(argv[1], "daytime");
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
    tcp::resolver::iterator end;

    tcp::socket socket(io_service);
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
      boost::array<char, 128> buf;
      asio::error error;

      size_t len = socket.read_some(
          asio::buffer(buf), asio::assign_error(error));

      if (error == asio::error::eof)
        break; // Connection closed cleanly by peer.
      else if (error)
        throw error; // Some other error.

      std::cout.write(buf.data(), len);
    }
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
