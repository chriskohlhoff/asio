#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

class accept_handler
{
public:
  accept_handler(demuxer& d)
    : demuxer_(d),
      acceptor_(d, inet_address_v4(32123)),
      peer_(d),
      handler_(boost::bind(&accept_handler::handle_accept, this, _1)),
      accept_count_(0)
  {
    acceptor_.async_accept_address(peer_, peer_address_, handler_);
  }

  void handle_accept(const socket_error& error)
  {
    if (!error)
    {
      if (peer_address_.good())
      {
        std::cout << "Accepted connection from " << peer_address_.host_name()
          << " (" << peer_address_.host_addr_str() << ")\n";
      }

      if (++accept_count_ < 10)
      {
        peer_.close();
        acceptor_.async_accept_address(peer_, peer_address_, handler_);
      }
    }
  }

private:
  demuxer& demuxer_;
  socket_acceptor acceptor_;
  stream_socket peer_;
  inet_address_v4 peer_address_;
  boost::function1<void, const socket_error&> handler_;
  int accept_count_;
};

int main()
{
  try
  {
    demuxer d;
    accept_handler a(d);
    d.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
