#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

class connect_handler
{
public:
  connect_handler(demuxer& d)
    : demuxer_(d),
      timer_queue_(d),
      connector_(d),
      socket_(d)
  {
    connector_.async_connect(socket_, inet_address_v4(32123, "localhost"),
        boost::bind(&connect_handler::handle_connect, this, _1));

    boost::xtime expiry_time;
    boost::xtime_get(&expiry_time, boost::TIME_UTC);
    expiry_time.sec += 5;
    timer_queue_.schedule_timer(expiry_time,
        boost::bind(&connect_handler::handle_timeout, this));
  }

  void handle_timeout()
  {
    connector_.close();
  }

  void handle_connect(const socket_error& error)
  {
    if (error)
    {
      std::cout << "Connect error: " << error.what() << "\n";
    }
    else
    {
      std::cout << "Successful connection\n";
    }
  }

private:
  demuxer& demuxer_;
  timer_queue timer_queue_;
  socket_connector connector_;
  stream_socket socket_;
};

int main()
{
  try
  {
    demuxer d;
    socket_acceptor a(d, inet_address_v4(32123), 1);
    connect_handler ch1(d);
    connect_handler ch2(d);
    connect_handler ch3(d);
    connect_handler ch4(d);
    connect_handler ch5(d);
    connect_handler ch6(d);
    connect_handler ch7(d);
    connect_handler ch8(d);
    connect_handler ch9(d);
    d.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
