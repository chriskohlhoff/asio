#include <ctime>
#include <iostream>
#include <string>
#include <asio.hpp>

int main()
{
  try
  {
    asio::demuxer demuxer;

    asio::datagram_socket socket(demuxer,
        asio::ipv4::udp::endpoint(13));

    for (;;)
    {
      char recv_buf[1];
      asio::ipv4::udp::endpoint remote_endpoint;
      asio::error error;
      socket.receive_from(
          asio::buffer(recv_buf, sizeof(recv_buf)),
          0, remote_endpoint, asio::assign_error(error));
      if (error && error != asio::error::message_size)
        throw error;

      using namespace std; // For time_t, time and ctime.
      time_t now = time(0);
      std::string msg = ctime(&now);

      socket.send_to(asio::buffer(msg.c_str(), msg.length()),
          0, remote_endpoint, asio::ignore_error());
    }
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
