#include <asio.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include "logger.hpp"
#include "stream_socket_service.hpp"

typedef asio::basic_stream_socket<
  services::stream_socket_service<asio::ipv4::tcp> > debug_stream_socket;

char read_buffer[1024];

void read_handler(const asio::error& e,
    std::size_t bytes_transferred, debug_stream_socket* s)
{
  if (!e)
  {
    std::cout.write(read_buffer, bytes_transferred);

    s->async_read_some(asio::buffer(read_buffer),
        boost::bind(read_handler, asio::placeholders::error,
          asio::placeholders::bytes_transferred, s));
  }
}

void connect_handler(const asio::error& e, debug_stream_socket* s)
{
  if (!e)
  {
    s->async_read_some(asio::buffer(read_buffer),
        boost::bind(read_handler, asio::placeholders::error,
          asio::placeholders::bytes_transferred, s));
  }
  else
  {
    std::cerr << e << std::endl;
  }
}

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: daytime_client <host>" << std::endl;
      return 1;
    }

    asio::io_service io_service;

    // Set the name of the file that all logger instances will use.
    services::logger logger(io_service, "");
    logger.use_file("log.txt");

    // Resolve the address corresponding to the given host.
    asio::ipv4::host_resolver host_resolver(io_service);
    asio::ipv4::host host;
    host_resolver.get_host_by_name(host, argv[1]);
    asio::ipv4::tcp::endpoint remote_endpoint(13, host.address(0));

    // Start an asynchronous connect.
    debug_stream_socket socket(io_service);
    socket.async_connect(remote_endpoint,
        boost::bind(connect_handler,
          asio::placeholders::error, &socket));

    // Run the io_service until all operations have finished.
    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
