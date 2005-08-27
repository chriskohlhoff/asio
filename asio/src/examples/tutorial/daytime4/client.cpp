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

    asio::datagram_socket socket(demuxer, asio::ipv4::udp::endpoint(0));

    asio::ipv4::host_resolver host_resolver(demuxer);
    asio::ipv4::host host;
    host_resolver.get_host_by_name(host, argv[1]);
    asio::ipv4::udp::endpoint receiver_endpoint(13, host.addresses[0]);

    char send_buf[1] = { 0 };
    socket.send_to(send_buf, sizeof(send_buf), receiver_endpoint);

    char recv_buf[128];
    asio::ipv4::udp::endpoint sender_endpoint;
    size_t len = socket.receive_from(recv_buf,
        sizeof(recv_buf), sender_endpoint);
    std::cout.write(recv_buf, len);
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
