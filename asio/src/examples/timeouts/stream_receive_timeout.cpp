#include "asio.hpp"
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>

using namespace asio;

class stream_handler
{
public:
  stream_handler(io_service& ios)
    : io_service_(ios),
      timer_(ios),
      acceptor_(ios, ipv4::tcp::endpoint(32123)),
      socket_(ios)
  {
    acceptor_.async_accept(socket_,
        boost::bind(&stream_handler::handle_accept, this,
          asio::placeholders::error));
  }

  void handle_accept(const error& err)
  {
    if (err)
    {
      std::cout << "Accept error: " << err << "\n";
    }
    else
    {
      std::cout << "Successful accept\n";

      socket_.async_read_some(asio::buffer(buf_, sizeof(buf_)),
          boost::bind(&stream_handler::handle_recv, this,
            asio::placeholders::error));
      timer_.expires_from_now(boost::posix_time::seconds(5));
      timer_.async_wait(boost::bind(&stream_handler::close, this));
    }
  }

  void handle_recv(const error& err)
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
  ipv4::tcp::acceptor acceptor_;
  ipv4::tcp::socket socket_;
  char buf_[1024];
};

void connect_handler()
{
  std::cout << "Successful connect\n";
}

int main()
{
  try
  {
    io_service ios;

    stream_handler sh(ios);

    ipv4::tcp::socket s(ios);
    s.async_connect(ipv4::tcp::endpoint(32123, ipv4::address::loopback()),
        boost::bind(connect_handler));

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
