#include <iostream>
#include <sstream>
#include <string>
#include <boost/bind.hpp>
#include "asio.hpp"

const short multicast_port = 30001;
const std::string multicast_addr = "225.0.0.1";
const int max_message_count = 10;

class sender
{
public:
  sender(asio::demuxer& d)
    : socket_(d, asio::ipv4::udp::endpoint(0)),
      timer_(d),
      message_count_(0)
  {
    std::ostringstream os;
    os << "Message " << message_count_++;
    message_ = os.str();

    asio::ipv4::udp::endpoint target(multicast_port, multicast_addr);
    socket_.async_sendto(message_.data(), message_.length(), target,
        boost::bind(&sender::handle_sendto, this, asio::arg::error));
  }

  void handle_sendto(const asio::error& error)
  {
    if (!error && message_count_ < max_message_count)
    {
      timer_.expiry(asio::time::now() + 1);
      timer_.async_wait(
          boost::bind(&sender::handle_timeout, this, asio::arg::error));
    }
  }

  void handle_timeout(const asio::error& error)
  {
    if (!error)
    {
      std::ostringstream os;
      os << "Message " << message_count_++;
      message_ = os.str();

      asio::ipv4::udp::endpoint target(multicast_port, multicast_addr);
      socket_.async_sendto(message_.data(), message_.length(), target,
          boost::bind(&sender::handle_sendto, this, asio::arg::error));
    }
  }

private:
  asio::datagram_socket socket_;
  asio::timer timer_;
  int message_count_;
  std::string message_;
};

int main(int argc, char* argv[])
{
  try
  {
    asio::demuxer d;
    sender s(d);
    d.run();
  }
  catch (asio::error& e)
  {
    std::cerr << e << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
