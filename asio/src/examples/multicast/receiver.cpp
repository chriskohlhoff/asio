#include <iostream>
#include <string>
#include "asio.hpp"
#include "boost/bind.hpp"

const short multicast_port = 30001;
const std::string multicast_addr = "225.0.0.1";

class receiver
{
public:
  receiver(asio::io_service& io_service)
    : socket_(io_service)
  {
    // Create the socket so that multiple may be bound to the same address.
    socket_.open(asio::ipv4::udp());
    socket_.set_option(asio::ipv4::udp::socket::reuse_address(true));
    socket_.bind(asio::ipv4::udp::endpoint(multicast_port));

    // Join the multicast group.
    socket_.set_option(
        asio::ipv4::multicast::add_membership(multicast_addr));

    socket_.async_receive_from(
        asio::buffer(data_, max_length), 0, sender_endpoint_,
        boost::bind(&receiver::handle_receive_from, this,
          asio::placeholders::error,
          asio::placeholders::bytes_transferred));
  }

  void handle_receive_from(const asio::error& error, size_t bytes_recvd)
  {
    if (!error)
    {
      std::cout.write(data_, bytes_recvd);
      std::cout << std::endl;

      socket_.async_receive_from(
          asio::buffer(data_, max_length), 0, sender_endpoint_,
          boost::bind(&receiver::handle_receive_from, this,
            asio::placeholders::error,
            asio::placeholders::bytes_transferred));
    }
  }

private:
  asio::ipv4::udp::socket socket_;
  asio::ipv4::udp::endpoint sender_endpoint_;
  enum { max_length = 1024 };
  char data_[max_length];
};

int main(int argc, char* argv[])
{
  try
  {
    asio::io_service io_service;
    receiver r(io_service);
    io_service.run();
  }
  catch (asio::error& e)
  {
    std::cerr << e << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
