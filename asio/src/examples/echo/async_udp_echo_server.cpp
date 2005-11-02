#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include "asio.hpp"

class server
{
public:
  server(asio::demuxer& d, short port)
    : demuxer_(d),
      socket_(d, asio::ipv4::udp::endpoint(port))
  {
    socket_.async_receive_from(asio::buffer(data_, max_length), 0,
        sender_endpoint_,
        boost::bind(&server::handle_receive_from, this,
          asio::placeholders::error, asio::placeholders::bytes_transferred));
  }

  void handle_receive_from(const asio::error& error, size_t bytes_recvd)
  {
    if (!error && bytes_recvd > 0)
    {
      socket_.async_send_to(asio::buffer(data_, bytes_recvd), 0,
          sender_endpoint_,
          boost::bind(&server::handle_send_to, this,
            asio::placeholders::error, asio::placeholders::bytes_transferred));
    }
    else
    {
      socket_.async_receive_from(asio::buffer(data_, max_length), 0,
          sender_endpoint_,
          boost::bind(&server::handle_receive_from, this,
            asio::placeholders::error, asio::placeholders::bytes_transferred));
    }
  }

  void handle_send_to(const asio::error& error, size_t bytes_sent)
  {
    socket_.async_receive_from(asio::buffer(data_, max_length), 0,
        sender_endpoint_,
        boost::bind(&server::handle_receive_from, this,
          asio::placeholders::error, asio::placeholders::bytes_transferred));
  }

private:
  asio::demuxer& demuxer_;
  asio::datagram_socket socket_;
  asio::ipv4::udp::endpoint sender_endpoint_;
  enum { max_length = 1024 };
  char data_[max_length];
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: async_udp_echo_server <port>\n";
      return 1;
    }

    asio::demuxer d;

    using namespace std; // For atoi.
    server s(d, atoi(argv[1]));

    d.run();
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
