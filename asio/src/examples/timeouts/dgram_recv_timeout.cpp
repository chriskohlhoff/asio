#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

class dgram_handler
{
public:
  dgram_handler(demuxer& d)
    : demuxer_(d),
      timer_(d),
      socket_(d, ipv4::udp::endpoint(32124))
  {
    socket_.async_recvfrom(data_, max_length, sender_endpoint_,
        boost::bind(&dgram_handler::handle_recvfrom, this, asio::arg::error,
          asio::arg::bytes_recvd));

    timer_.expiry(asio::time::now() + 5);
    timer_.async_wait(boost::bind(&dgram_socket::close, &socket_));
  }

  void handle_recvfrom(const error& err, size_t length)
  {
    if (err)
    {
      std::cout << "Receive error: " << err << "\n";
    }
    else
    {
      std::cout << "Successful receive\n";
    }
  }

private:
  demuxer& demuxer_;
  timer timer_;
  dgram_socket socket_;
  ipv4::udp::endpoint sender_endpoint_;
  enum { max_length = 512 };
  char data_[max_length];
};

int main()
{
  try
  {
    demuxer d;
    dgram_handler dh(d);
    d.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
