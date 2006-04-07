#include <ctime>
#include <iostream>
#include <string>
#include <asio.hpp>

int main()
{
  try
  {
    asio::io_service io_service;

    asio::ipv4::udp::socket socket(io_service,
        asio::ipv4::udp::endpoint(13));

    for (;;)
    {
      char recv_buf[1];
      asio::ipv4::udp::endpoint remote_endpoint;
      asio::error error;
      socket.receive_from(
          asio::buffer(recv_buf, sizeof(recv_buf)),
          remote_endpoint, 0, asio::assign_error(error));
      if (error && error != asio::error::message_size)
        throw error;

      using namespace std; // For time_t, time and ctime.
      time_t now = time(0);
      std::string msg = ctime(&now);

      socket.send_to(asio::buffer(msg.c_str(), msg.length()),
          remote_endpoint, 0, asio::ignore_error());
    }
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
