#include "asio.hpp"
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>

using namespace asio;

class accept_handler
{
public:
  accept_handler(io_service& ios)
    : io_service_(ios),
      timer_(ios),
      acceptor_(ios, ip::tcp::endpoint(ip::tcp::v4(), 32123)),
      socket_(ios)
  {
    acceptor_.async_accept(socket_,
        boost::bind(&accept_handler::handle_accept, this,
          asio::placeholders::error));

    timer_.expires_from_now(boost::posix_time::seconds(5));
    timer_.async_wait(boost::bind(&accept_handler::close, this));
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
    }
  }

  void close()
  {
    acceptor_.close();
  }

private:
  io_service& io_service_;
  deadline_timer timer_;
  ip::tcp::acceptor acceptor_;
  ip::tcp::socket socket_;
};

int main()
{
  try
  {
    io_service ios;
    accept_handler ah(ios);
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
