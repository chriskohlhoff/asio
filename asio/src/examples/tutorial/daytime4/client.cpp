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

    asio::ip::udp::resolver resolver(io_service);
    asio::ip::udp::resolver::query query(
        asio::ip::udp::v4(), argv[1], "daytime");
    asio::ip::udp::endpoint receiver_endpoint = *resolver.resolve(query);

    asio::ip::udp::socket socket(io_service);
    socket.open(asio::ip::udp::v4());

    char send_buf[1] = { 0 };
    socket.send_to(
        asio::buffer(send_buf, sizeof(send_buf)),
        receiver_endpoint);

    char recv_buf[128];
    asio::ip::udp::endpoint sender_endpoint;
    size_t len = socket.receive_from(
        asio::buffer(recv_buf, sizeof(recv_buf)),
        sender_endpoint);
    std::cout.write(recv_buf, len);
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
