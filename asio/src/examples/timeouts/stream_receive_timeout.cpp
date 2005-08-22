#include "asio.hpp"
#include <boost/bind.hpp>
#include <iostream>

using namespace asio;

class stream_handler
{
public:
  stream_handler(demuxer& d)
    : demuxer_(d),
      timer_(d),
      acceptor_(d, ipv4::tcp::endpoint(32123)),
      socket_(d)
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

      socket_.async_read(buf_, sizeof(buf_),
          boost::bind(&stream_handler::handle_recv, this,
            asio::placeholders::error));
      timer_.expiry(asio::time::now() + 5);
      timer_.async_wait(boost::bind(&stream_socket::close, &socket_));
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

private:
  demuxer& demuxer_;
  timer timer_;
  socket_acceptor acceptor_;
  stream_socket socket_;
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
    demuxer d;

    stream_handler sh(d);

    stream_socket s(d);
    socket_connector c(d);
    c.async_connect(s, ipv4::tcp::endpoint(32123, ipv4::address::loopback()),
        boost::bind(connect_handler));

    d.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
