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
      socket_(d, ipv4::address(32124))
  {
    socket_.async_recvfrom(data_, max_length, sender_address_,
        boost::bind(&dgram_handler::handle_recvfrom, this, asio::arg::error,
          asio::arg::bytes_recvd));

    timer_.set(timer::from_now, 5);
    timer_.async_wait(boost::bind(&dgram_handler::handle_timeout, this));
  }

  void handle_timeout()
  {
    socket_.close();
  }

  void handle_recvfrom(const socket_error& error, size_t length)
  {
    if (error)
    {
      std::cout << "Receive error: " << error.message() << "\n";
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
  ipv4::address sender_address_;
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
