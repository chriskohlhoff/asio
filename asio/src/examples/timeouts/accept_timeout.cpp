#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

class accept_handler
{
public:
  accept_handler(demuxer& d)
    : demuxer_(d),
      timer_(d),
      acceptor_(d, ipv4::address(32123)),
      socket_(d)
  {
    acceptor_.async_accept(socket_,
        boost::bind(&accept_handler::handle_accept, this, asio::arg::error));

    timer_.set(timer::from_now, 5);
    timer_.async_wait(boost::bind(&socket_acceptor::close, &acceptor_));
  }

  void handle_accept(const socket_error& error)
  {
    if (error)
    {
      std::cout << "Accept error: " << error.message() << "\n";
    }
    else
    {
      std::cout << "Successful accept\n";
    }
  }

private:
  demuxer& demuxer_;
  timer timer_;
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
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
