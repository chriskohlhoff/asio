#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

class dgram_handler
{
public:
  dgram_handler(demuxer& d)
    : demuxer_(d),
      socket_(d, inet_address_v4(12346))
  {
    socket_.async_recvfrom(data_, max_length, sender_address_,
        boost::bind(&dgram_handler::handle_recvfrom, this, _1, _2));
  }

  void handle_recvfrom(const socket_error& error, size_t length)
  {
    if (!error && length > 0)
    {
      socket_.async_sendto(data_, length, sender_address_,
          boost::bind(&dgram_handler::handle_sendto, this, _1, _2));
    }
  }

  void handle_sendto(const socket_error& error, size_t length)
  {
    if (!error)
    {
      socket_.async_recvfrom(data_, max_length, sender_address_,
          boost::bind(&dgram_handler::handle_recvfrom, this, _1, _2));
    }
  }

private:
  demuxer& demuxer_;
  dgram_socket socket_;
  inet_address_v4 sender_address_;
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
