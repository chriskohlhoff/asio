#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

class dgram_handler
{
public:
  dgram_handler(demuxer& d)
    : demuxer_(d),
      timer_queue_(d),
      socket_(d, inet_address_v4(32124))
  {
    socket_.async_recvfrom(data_, max_length, sender_address_,
        boost::bind(&dgram_handler::handle_recvfrom, this, _1, _2));

    boost::xtime expiry_time;
    boost::xtime_get(&expiry_time, boost::TIME_UTC);
    expiry_time.sec += 5;
    timer_queue_.schedule_timer(expiry_time,
        boost::bind(&dgram_handler::handle_timeout, this));
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
  timer_queue timer_queue_;
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
