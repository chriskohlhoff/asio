#include "asio.hpp"
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>

using namespace asio;

class datagram_handler
{
public:
  datagram_handler(io_service& ios)
    : io_service_(ios),
      timer_(ios),
      socket_(ios, ipv4::udp::endpoint(32124))
  {
    socket_.async_receive_from(
        asio::buffer(data_, max_length), sender_endpoint_,
        boost::bind(&datagram_handler::handle_receive_from, this,
          asio::placeholders::error,
          asio::placeholders::bytes_transferred));

    timer_.expires_from_now(boost::posix_time::seconds(5));
    timer_.async_wait(boost::bind(&datagram_handler::close, this));
  }

  void handle_receive_from(const error& err, size_t length)
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

  void close()
  {
    socket_.close();
  }

private:
  io_service& io_service_;
  deadline_timer timer_;
  ipv4::udp::socket socket_;
  ipv4::udp::endpoint sender_endpoint_;
  enum { max_length = 512 };
  char data_[max_length];
};

int main()
{
  try
  {
    io_service ios;
    datagram_handler dh(ios);
    ios.run();
  }
  catch (asio::error& e)
  {
    std::cerr << "Exception: " << e << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
