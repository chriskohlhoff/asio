#include "asio.hpp"
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
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

      socket_.async_read(asio::buffer(buf_, sizeof(buf_)),
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
  demuxer& demuxer_;
  deadline_timer timer_;
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
    s.async_connect(ipv4::tcp::endpoint(32123, ipv4::address::loopback()),
        boost::bind(connect_handler));

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
