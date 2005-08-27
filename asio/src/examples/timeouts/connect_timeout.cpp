#include "asio.hpp"
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>

using namespace asio;

class connect_handler
{
public:
  connect_handler(demuxer& d)
    : demuxer_(d),
      timer_(d),
      connector_(d),
      socket_(d)
  {
    connector_.async_connect(socket_,
        ipv4::tcp::endpoint(32123, ipv4::address::loopback()),
        boost::bind(&connect_handler::handle_connect, this,
          asio::placeholders::error));

    timer_.expires_from_now(boost::posix_time::seconds(5));
    timer_.async_wait(boost::bind(&socket_connector::close, &connector_));
  }

  void handle_connect(const error& err)
  {
    if (err)
    {
      std::cout << "Connect error: " << err << "\n";
    }
    else
    {
      std::cout << "Successful connection\n";
    }
  }

private:
  demuxer& demuxer_;
  deadline_timer timer_;
  socket_connector connector_;
  stream_socket socket_;
};

int main()
{
  try
  {
    demuxer d;
    socket_acceptor a(d, ipv4::tcp::endpoint(32123), 1);

    // Make lots of connections so that at least some of them will block.
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
