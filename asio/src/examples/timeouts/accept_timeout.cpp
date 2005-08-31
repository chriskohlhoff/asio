#include "asio.hpp"
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <iostream>

using namespace asio;

class accept_handler
{
public:
  accept_handler(demuxer& d)
    : demuxer_(d),
      timer_(d),
      acceptor_(d, ipv4::tcp::endpoint(32123)),
      socket_(d)
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
  demuxer& demuxer_;
  deadline_timer timer_;
  socket_acceptor acceptor_;
  stream_socket socket_;
};

int main()
{
  try
  {
    demuxer d;
    accept_handler ah(d);
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
